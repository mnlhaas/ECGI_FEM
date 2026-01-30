import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P


class ZeroMean(nn.Module):
    """Enforce zero mean kernels for each output channel"""
    def forward(self, x):
        return x - x.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(2)

class FemConvolution(nn.Module):
    def __init__(self, num_experts_in, num_experts_out, size_kernel, lumped, approx):
        """Temporal convolution using 1D FEM matrix"""
        super().__init__()
        
        self.lumped = lumped
        self.K = size_kernel
        self.conv_param = nn.Conv1d(in_channels=num_experts_in, out_channels=num_experts_out,kernel_size=self.K, padding=size_kernel//2+1, bias=False)
        if approx:
            sten = torch.tensor([1, 4, 1]).unsqueeze(0).unsqueeze(0)
            self.register_buffer("sten", sten)
            if lumped:
                self.convolution = self.conv_approx_lump
                self.transpose = self.trans_approx_lump
            else:
                self.convolution = self.conv_approx
                self.transpose = self.trans_approx
        else:
            self.convolution = self.conv
            self.transpose = self.trans
    
    
    def conv(self, x, dt, fem_matrices):
        fem_matrices_K = fem_matrices[self.K]
        weight = self.conv_param.weight
        x_fem = torch.einsum('tkh, bih -> bikt', fem_matrices_K, x)
        x_out = torch.einsum('oik, bikt -> bot', weight, x_fem)
        return dt/6 * x_out
            
    def trans(self, x, dt, fem_matrices):
        fem_matrices_K = fem_matrices[self.K]
        weight = self.conv_param.weight
        x_fem_T = torch.einsum('oik, bot -> bikt', weight, x)
        x_T = torch.einsum('tkh, bikt -> bih', fem_matrices_K, x_fem_T)
        return dt/6 * x_T
            
    def conv_approx(self, x, dt, fem_matrices=None):
        weight = self.conv_param.weight
        base = F.conv1d(x, weight, bias=None, dilation=self.conv_param.dilation, padding=self.conv_param.padding, groups=self.conv_param.groups, stride=self.conv_param.stride)
        mass = F.conv1d(base, self.sten.expand(base.shape[1],1,3), padding=0, groups=base.shape[1])
        return dt/6 * mass
    
    def conv_approx_lump(self, x, dt, fem_matrices=None):
        weight = self.conv_param.weight
        base = F.conv1d(x, weight, bias=None, dilation=self.conv_param.dilation, padding=self.conv_param.padding, groups=self.conv_param.groups, stride=self.conv_param.stride)
        return dt * base[:,:,1:-1]
    
    def trans_approx(self, x, dt, fem_matrices=None):
        weight = self.conv_param.weight
        x_new = dt/6 * F.conv_transpose1d(x, self.sten.expand(x.shape[1],1,3), padding=0, groups=x.shape[1])
        return F.conv_transpose1d(x_new, weight, bias=None, dilation=self.conv_param.dilation, padding=self.conv_param.padding, groups=self.conv_param.groups, stride=self.conv_param.stride)
    
    def trans_approx_lump(self, x, dt, fem_matrices=None):
        bw, c, h = x.shape
        weight = self.conv_param.weight
        x_new = torch.zeros((bw, c, h + 2), device=x.device, dtype=x.dtype)
        x_new[:,:,1:-1] = dt * x
        return F.conv_transpose1d(x_new, weight, bias=None, dilation=self.conv_param.dilation, padding=self.conv_param.padding, groups=self.conv_param.groups, stride=self.conv_param.stride)

class L_Operator(nn.Module):
    def __init__(self, num_experts, size_kernels, eps_theta, lumped=False, approx=False):
        """" 
        Compute the linear operators L_i = (\epsilon u, \nabla_x u, k_i \ast_T u)
        """
        super().__init__()

        self.lumped = lumped
        self.approx = approx
        self.size_kernels = size_kernels
        self.num_experts = num_experts

        # list of convolutionnal layers
        self.conv_layers = nn.ModuleList()
        for j in range(len(num_experts) - 1):
            self.conv_layers.append(FemConvolution(num_experts[j], num_experts[j+1], size_kernels[j], lumped=self.lumped, approx=self.approx))
        P.register_parametrization(self.conv_layers[0].conv_param, "weight", ZeroMean())
            

        # cache the estimation of the spectral norm
        self.L = torch.tensor(1., requires_grad=True)
        self.eps_theta = nn.Parameter(torch.tensor(eps_theta)) 
        self.padding_total = sum([kernel_size//2 for kernel_size in self.size_kernels])
        
        self.fem_matrices = None
    

    def forward(self, x, proj_p1, Ks, dt, d):
        b, c, w, h = x.shape
        x = x / torch.sqrt(self.L) 
        
        #Compute temporal convolution
        x_time = x.permute(0, 2, 1, 3).reshape(b * w, c, h)
        for conv in self.conv_layers:
            x_time = conv.convolution(x_time, dt, self.fem_matrices)
                
        x_time = x_time.reshape(b, w, self.num_experts[-1], -1).permute(0, 2, 1, 3)
        
        #Compute spatial gradient
        x_space = (Ks@x).reshape([1,1,-1,d,h])
        x_space_int = torch.einsum('bckn,bcndt->bckdt', proj_p1, x_space).permute(dims=(0,3,1,2,4))
        
        #Interpolate all to the same space
        x = torch.concat([self.eps_theta.exp()*x.repeat(1,1,self.num_experts[-1],1,1),x_space_int.repeat(1,1,self.num_experts[-1],1,1),x_time.unsqueeze(1)], axis=1)
        
        return x

    def adjoint(self, x, proj_p1, Ks, M, M_inv, D, D_inv, dt):
        b, d, c, w, h = x.shape
        x = x / torch.sqrt(self.L) 
        
        x_plain = torch.sum(x[:,0], axis=1, keepdims=True)
        x_space = torch.einsum('bckn,bdcnt->bdckt', M, x[:,1:d-1])
        x_space = torch.sum(x_space, axis=2, keepdims=True)
        x_time = torch.einsum('bckt,bcnt->bcnk', D, x[:,d-1])
        x_time = x_time.permute(0, 2, 1, 3).reshape(b * w, c, h)
        
        #Compute temporal transposed convolution
        for conv in reversed(self.conv_layers):
            x_time = conv.transpose(x_time, dt, self.fem_matrices)
                
        x_time = x_time.reshape(b, w, 1, -1).permute(0, 2, 1, 3)
        
        #Compute spatial gradient
        x_space = x_space.permute(dims=(0,2,3,1,4))
        x_space = torch.einsum('bckn,bckdt->bcndt', proj_p1, x_space).reshape([1,1,-1,h])
        x_space = torch.permute(Ks, (0,1,3,2))@x_space
        
        #Summarize all dimensions
        x = self.eps_theta.exp()*x_plain + M_inv@x_space + torch.einsum('bckt,bcnt->bcnk', D_inv, x_time)

        return x

    def spectral_norm(self, space_nodes, time_steps, proj_p1, Ks, M, M_inv, D, D_inv, dt, d, n_steps=500, tol=1e-4):
        """ 
        Compute the spectral norm of the linear operators with the power method for n_steps steps
        """
        self.L = torch.tensor(1., device=self.conv_layers[0].conv_param.weight.device)
        u = torch.empty((1, 1, space_nodes, time_steps), device= self.conv_layers[0].conv_param.weight.device, dtype=Ks.dtype).normal_()
        with torch.no_grad():
            for i in range(n_steps):
                v = self.adjoint(self.forward(u, proj_p1, Ks, dt, d), proj_p1, Ks, M, M_inv, D, D_inv, dt)
                
                lambda_new = torch.sum(u * v)
                v_norm = torch.linalg.norm(v)
                if v_norm == 0:
                    break
                u = v / v_norm
                
                if i > 0:
                    rel_change = torch.abs(lambda_new - lambda_old) / torch.abs(lambda_new)
                    if rel_change < tol:
                        self.power_iteration = i
                        break
                    
                lambda_old = lambda_new
            
        self.L = torch.linalg.norm(self.adjoint(self.forward(u, proj_p1, Ks, dt, d), proj_p1, Ks, M, M_inv, D, D_inv, dt))

        return self.L
    
    def _fem_conv_matrix(self, K, S, s, device):
        # Effective size of the tridiagonal block
        size_eff = min(K, K//2 + 1 + s, K - (s - (S-1 - K//2)))

        offset_row = max(0, K//2 - s)
        offset_col = max(0, s - K//2)
        
        # Number of nonzeros in tridiagonal
        nnz = 3*size_eff - 2 
        indices = torch.empty((2, nnz), dtype=torch.long, device=device)
        values = torch.empty(nnz, device=device)

        # Main diagonal
        main_idx = torch.arange(size_eff, device=device)
        indices[0, :size_eff] = main_idx + offset_row
        indices[1, :size_eff] = main_idx + offset_col
        values[:size_eff] = 4
        values[0] = values[size_eff-1] = 2

        # Upper diagonal
        upper_idx = torch.arange(size_eff - 1, device=device)
        indices[0, size_eff:size_eff*2 - 1] = upper_idx + offset_row
        indices[1, size_eff:size_eff*2 - 1] = upper_idx + 1 + offset_col
        values[size_eff:size_eff*2 - 1] = 1

        # Lower diagonal
        indices[0, size_eff*2 - 1:] = upper_idx + 1 + offset_row
        indices[1, size_eff*2 - 1:] = upper_idx + offset_col
        values[size_eff*2 - 1:] = 1

        return torch.sparse_coo_tensor(indices, values, size=(K, S), device=device).coalesce().to_dense()
    
    def _fem_conv_matrix_lumped(self, K, S, s, device):
        # Effective size of the tridiagonal block
        size_eff = min(K, K//2 + 1 + s, K - (s - (S-1 - K//2)))

        offset_row = max(0, K//2 - s)
        offset_col = max(0, s - K//2)

        # Number of nonzeros in tridiagonal
        indices = torch.empty((2, size_eff), dtype=torch.long, device=device)
        values = torch.empty(size_eff, device=device)

        # Main diagonal
        main_idx = torch.arange(size_eff, device=device)
        indices[0, :] = main_idx + offset_row
        indices[1, :] = main_idx + offset_col
        values[:] = 6
        values[0] = values[-1] = 3

        return torch.sparse_coo_tensor(indices, values, size=(K, S), device=device).coalesce().to_dense()
    
    def precompute_fem_matrix(self, S, device, dtype):
        self.fem_matrices = {}
        unique_kernel_size = list(dict.fromkeys(self.size_kernels))
        assem_fem_matrix = self._fem_conv_matrix_lumped if self.lumped else self._fem_conv_matrix
        for _, K in enumerate(unique_kernel_size):
            fem_matrices_for_K = [assem_fem_matrix(K, S, i, device).to(dtype=dtype) for i in range(S)]
            self.fem_matrices[K] = torch.stack(fem_matrices_for_K, dim=0)

    def get_filters(self, proj_p1, Ks, dt, d):
        """Plot temporal filters at their nodes"""
        time_steps = 4 * self.padding_total + 1
        dirac = torch.zeros((1, 1) + (Ks.shape[-1], 4 * self.padding_total + 1)).to(Ks.device, Ks.dtype)
        dirac[0, 0, :, 2 * self.padding_total] = 1
        if not self.approx:
            self.precompute_fem_matrix(time_steps, dirac.device, dirac.dtype)
        kernel = self.forward(dirac, proj_p1, Ks, dt, d)[:,-1,:,0,self.padding_total:3*self.padding_total+1]
        return kernel