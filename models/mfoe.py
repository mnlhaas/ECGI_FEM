import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from models.l_operator import L_Operator
from models.optimization import AGDR, proj_l1_channel, L2_squared, L2_squared_torso

if torch.is_grad_enabled():
    from torchdeq import get_deq

class MFoE_temp(torch.nn.Module):
    """
    Multivariate FoE model 
    """
    def __init__(self, model_params, l_op_params, fw_param, bw_param):
        super(MFoE_temp, self).__init__()
        if model_params['problem'] == 'denoise':
            self.reconstruct = self.reconstruct_denoise
            
        elif model_params['problem'] == 'inverse':
            self.reconstruct = self.reconstruct_inverse
            
        self.l_op = L_Operator(**l_op_params) 
        self.nb_groups = l_op_params['num_experts'][-1]
        self.groupsize = model_params['groupsize']
        self.d = self.groupsize - 2
        
        self.convex = model_params['convex']
        self.lamb = nn.Parameter(torch.tensor(model_params['lamb_init']))
        self.Q_param = nn.Parameter(torch.rand(self.nb_groups, self.groupsize, self.groupsize) - 0.5)
        self.taus = nn.Parameter(model_params['lamb_init']/2*torch.ones(1, 1, self.nb_groups, 1, 1))
        self.scale = 0.999
        
        self.slope = model_params['scaling']
        self.mu = nn.Sequential(*[nn.Linear(1, self.nb_groups), nn.ReLU(), nn.Linear(self.nb_groups, self.nb_groups),nn.ReLU(), nn.Linear(self.nb_groups, self.nb_groups)])
        self.eps_omega = nn.Parameter(torch.tensor(model_params['eps_omega'])) 
        
        self.param_fw = fw_param
        self.param_bw = bw_param

        self.num_params = sum(p.numel() for p in self.parameters())

        # Parameters to cache
        self.scaling = None
        self.Q = None
        self.Q_norms = None

    def forward(self, x_noisy, sigma, proj_p1, Ks, M, M_inv, D, D_inv, dt, A, L_data_fid):

        # update spectral norm of the convolutional layer
        space_nodes = Ks.shape[-1]
        time_steps = x_noisy.shape[-1]
        if not self.l_op.approx:
            self.l_op.precompute_fem_matrix(time_steps, x_noisy.device, x_noisy.dtype)
        self.l_op.spectral_norm(space_nodes, time_steps, proj_p1, Ks, M, M_inv, D, D_inv, dt, self.d)

        # fixed point iteration
        def f(x):
            return x - self.reconstruct(x, x_noisy, sigma, proj_p1, Ks, M, M_inv, D, D_inv, dt, A, L_data_fid)[0]

        def f_solver(deq_func, x0, max_iter, tol, stop_mode, **solver_kwargs):
            z, self.fw_niter_max, self.fw_niter_mean = AGDR(x0.view(x_noisy.shape), x0.view(x_noisy.shape), proj_p1, Ks, M, M_inv, D, D_inv, dt, A, L_data_fid, self, sigma, **self.param_fw)
            return z.view(x0.shape), [], []

        deq = get_deq(f_max_iter=self.param_fw['max_iter'], f_tol=self.param_fw['tol'], b_solver='broyden',
                      b_max_iter=self.param_bw['max_iter'], b_tol=self.param_bw['tol'], ift=True, kwargs={'ls': True})

        deq.f_solver = f_solver
        z = deq(f, x_noisy)[0][-1]

        return z
    
    def get_scaling(self, sigma):
        if self.scaling is None:
            sigma = sigma[:, :, 0, 0]
            scaling = F.relu(self.mu(sigma*20 - 2)*0.05 + sigma) * self.slope + 1e-9
            scaling = scaling.view(-1, 1, self.nb_groups, 1, 1)
            return scaling
        else:
            return self.scaling
        
    def clear_cache(self):
        self.scaling = None
        self.Q = None
        self.Q_norms = None
        
    def orient(self, x):
        "Apply the matrix Q"
        if self.groupsize > 1:
            if self.Q is None:
                self.Q = self.Q_param / torch.max(torch.sum(self.Q_param.abs(), dim=2,keepdim=True), torch.tensor(1.0))
                self.Q_norms = torch.linalg.matrix_norm(
                    self.Q, ord=2, dim=(1, 2), keepdim=True)
            Q = self.Q / self.Q_norms**2
            return self.scale * torch.einsum('flg,bgfhw->blfhw', Q, x)
        else:
            return self.scale * x

    def unorient(self, x):
        "Apply the matrix Q^T"
        if self.groupsize > 1:
            return self.scale * torch.einsum('flg,bgfhw->blfhw', self.Q.transpose(1, 2), x)
        else:
            return self.scale * x

    def moreau(self, x):
        if self.groupsize > 1:
            grad = proj_l1_channel(x)
        else:
            grad = torch.clip(x, -1., 1.)
        cost = (x - grad).norm(dim=1, p=float('inf')) + (1/2)*grad.norm(dim=1, p=2)**2
        cost = cost.unsqueeze(1)
        
        # stability for Mosco convergence
        grad = grad + self.eps_omega.exp() * x
        cost = cost + self.eps_omega.exp()/2 * x.norm(dim=1, p=2, keepdim=True)
        return grad, cost

    def activation(self, x):
        grad_convex, cost_convex = self.moreau(x)
        if self.convex:
            grad_concave, cost_concave = 0., 0.
        else:
            taus = F.relu(self.taus).exp()
            grad_concave, cost_concave = self.moreau(self.orient(x) / taus)
            grad_concave = self.unorient(grad_concave)
            cost_concave = taus * cost_concave
            if self.groupsize > 1:
                cost_concave = self.Q_norms**2 * cost_concave
        grad = self.lamb.exp() * (grad_convex - grad_concave)
        cost = self.lamb.exp() * (cost_convex - cost_concave)
        return grad, cost
    
    def grad_cost(self, x, sigma, proj_p1, Ks, M, M_inv, D, D_inv, dt):
        Lx = self.l_op(x, proj_p1, Ks, dt, self.d)
        
        # x = torch.rand_like(x)
        # Lx = self.l_op(x, proj_p1, Ks, dt, self.d)
        # y = torch.rand_like(Lx)
        # Ly = self.l_op.adjoint(y, proj_p1, Ks, M, M_inv, D, D_inv, dt)
        
        # def scalar(u, v, M, D, Ks=None, Kt=None, dx=None, dt=None):
        #     diff = u
        #     tmp = torch.einsum('...ij,...ki->...kj', diff, M)
        #     tmp = torch.einsum('...ij,...kj->...ik', tmp, D)
        #     return torch.sqrt(torch.sum(v * tmp))
        
        # print(scalar(x,Ly, M, D) - scalar(y,Lx, M, D))
        
        # Applying nonlinearity
        scaling = self.get_scaling(sigma)
        Lx = Lx / scaling
        grad, cost = self.activation(Lx)
        grad = grad * scaling
        
        grad = self.l_op.adjoint(grad, proj_p1, Ks, M, M_inv, D, D_inv, dt)
        
        cost = cost * scaling**2
        # Lumped quadrature
        cost = (M.sum(-1).unsqueeze(-1).unsqueeze(-3) * D.sum(-2).unsqueeze(-2).unsqueeze(-3) * cost).sum(dim = (1,2,3,4))
        return grad, cost

    def reconstruct_denoise(self, x, x_noisy, sigma, proj_p1, Ks, M, M_inv, D, D_inv, dt, A=None, L_data_fid=None):
        grad, cost = self.grad_cost(x, sigma, proj_p1, Ks, M, M_inv, D, D_inv, dt)
        grad = 1. / (1. + self.lamb.exp()*(1+self.eps_omega.exp())) * (x - x_noisy + grad)
        cost = cost + (1/2)*L2_squared(x, x_noisy, M, D)
        return grad, cost
    
    def reconstruct_inverse(self, x, x_noisy, sigma, proj_p1, Ks, M, M_inv, D, D_inv, dt, A, L_data_fid):
        grad, cost = self.grad_cost(x, sigma, proj_p1, Ks, M, M_inv, D, D_inv, dt)
        grad = 1. / (L_data_fid/A.shape[0] + self.lamb.exp()*(1+self.eps_omega.exp())) * (1 / A.shape[0] * M_inv@torch.permute(A,(0,1,3,2))@(A@x-x_noisy) + grad) 
        cost = cost + (1/2)*L2_squared_torso(x, x_noisy, A, D)
        return grad, cost