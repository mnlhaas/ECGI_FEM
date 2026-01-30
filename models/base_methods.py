import numpy as np
from tqdm import trange

import cupy as cp
from cupy.linalg import cholesky, inv
from cupyx.scipy.linalg import solve_triangular
from cupyx.scipy.sparse import csr_matrix, diags
from cupyx.scipy.sparse.linalg import LinearOperator, cg, eigsh, splu

def L2_cp(u, v, M, D, Ks=None, Kt=None, dx=None, dt=None):
    diff = u - v
    return cp.sqrt(cp.sum(diff * (D@(M@diff).T).T))

def H1_cp(u, v, M, D, Ks, Kt, dx, dt):
    diff = u - v
    
    l2 = cp.sum(diff * (D@(M@diff).T).T)
    
    grad_s = Ks @ diff
    grad_s_weight = dx.flatten()[:, cp.newaxis] * (D @ grad_s.T).T
    grad_s_comb = cp.sum(grad_s * grad_s_weight)
    
    grad_t = (Kt @ diff.T).T
    grad_t_weight = dt * M @ grad_t
    grad_t_comb = cp.sum(grad_t*grad_t_weight)
        
    return cp.sqrt(l2 + grad_s_comb + grad_t_comb)

class Base_Methods:
    def __init__(self, config, device):
        self.config = config
        
        gpu_id = int(device.split(":")[1])
        cp.cuda.Device(gpu_id).use()
        
        if config['precision'] == "double":                
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.lumped = config['lumped']
        self.max_iter = config['max_iter']
        
        self.problem = config['problem']
        # Dataset parameters
        if config["reg"] == 'TV':
            self.optim = self.tv_primal_dual
            self.norm = config["norm"]
            
        elif config["reg"] == 'TIK':
            self.optim = self.tik_cg
            self.norm = config["norm"]

    ### Gradient Operators ###
    
    # Gradient and Gradient* for isotropic TV (l2)
    def grad_l2(self, u, lamb, Ks, Kt, N_0, d, int_op_space):
        grad_x = lamb[0] * cp.tile((Ks @ u).reshape([N_0, d, -1]), [d, 1, 1, 1]) #[Quadrature Points Space, Elements, Dimension, Time]
        grad_x_u = cp.stack([grad_x[:,:,:,:-1],grad_x[:,:,:,1:]], axis = 3) #[Quadrature Points Space, Elements, Dimension, Quadrature Points Time, Time]
        quadr_u_full = (int_op_space @ u).reshape([d, N_0, -1])
        grad_t_u = lamb[1] * cp.tile(cp.stack([(Kt @ quadr_u.T).T for quadr_u in quadr_u_full], axis=0)[..., cp.newaxis, cp.newaxis,:], [1,1,1,2,1])
        grad_xt_u = cp.concatenate([grad_x_u, grad_t_u], axis=2)
        return grad_xt_u

    def div_l2(self, p, lamb, Ks, Kt, N_0, d, dx, dt, M_weight, D_weight, int_op_space):
        p_w = dt/2 * dx.T[...,cp.newaxis, cp.newaxis, cp.newaxis] * p
        grad_x_u, grad_t_u = cp.split(p_w, [d], axis=2)
        div_quadr_u = cp.transpose(cp.sum(cp.squeeze(grad_t_u), axis = 2), (0,2,1))
        
        div_quadr_t_u = cp.stack([(Kt.T @ grad_quadr_t_u).T for grad_quadr_t_u in div_quadr_u], axis=0)
        div_t_u = int_op_space.T @ div_quadr_t_u.reshape([d*N_0, -1])
        
        grad_x_m_1 = cp.concatenate((grad_x_u[:,:,:,0], cp.zeros((d,N_0,d,1))), axis=3)
        grad_x_m_2 = cp.concatenate((cp.zeros((d,N_0,d,1)), grad_x_u[:,:,:,1]), axis=3)
        
        grad_x_m = (grad_x_m_1+grad_x_m_2).reshape([d, N_0*d, -1])
        div_x_u = sum([Ks.T @ grad_quadr_x_u for grad_quadr_x_u in grad_x_m]) 
        
        KT_p = lamb[1] * div_t_u + lamb[0] * div_x_u
        if self.lumped:
            return M_weight*D_weight*KT_p
        else:
            return M_weight.solve(D_weight.solve(KT_p.T).T)

    # Gradient and Gradient* for anisotropic TV (l1)
    def grad_space(self, u, lamb_s, Ks, N_0, d):
        grad_x_u = lamb_s * (Ks@u).reshape([N_0,d,-1]) #[Elements, Dimension, Time Points]
        return grad_x_u
                
    def grad_time(self, u, lamb_t, Kt):
        grad_t_u = lamb_t * (Kt @ u.T).T #[Points, Time Elements]
        return grad_t_u

    def grad_l1(self, u, lamb, Ks, Kt, N_0, d, int_op_space=None):
        return (self.grad_space(u, lamb[0], Ks, N_0, d) ,self.grad_time(u, lamb[1], Kt))

    def div_space(self, p, lamb_s, Ks, dx, N_0, d, M_weight):
        p_w = dx.sum(1)[:, cp.newaxis, cp.newaxis] *p
        div_x_u = Ks.T @ p_w.reshape([N_0*d,-1])
        KT_p = lamb_s * div_x_u
        if self.lumped:
            return M_weight*KT_p
        else:
            return M_weight.solve(KT_p)
        
    def div_time(self, p, lamb_t, Kt, D_weight, dt):
        p_w = dt * p
        div_t_u = (Kt.T @ p_w.T).T
        KT_p = lamb_t * div_t_u
        if self.lumped:
            return D_weight*KT_p
        else:
            return D_weight.solve(KT_p.T).T

    def div_l1(self, p, lamb, Ks, Kt, N_0, d, dx, dt, M_weight, D_weight, int_op_space=None):
        return self.div_space(p[0], lamb[0], Ks, dx, N_0, d, M_weight) + self.div_time(p[1], lamb[1], Kt, D_weight, dt)

    # Spatiotemporal Laplacian for first-order Tikhonov
    def space_time_laplacian(self, u, lamb, Ks, Kt, N_0, d, dx, dt, M_weight, D_weight):
        u_grad = self.grad_l1(u, lamb, Ks, Kt, N_0, d)
        u_lap = self.div_l1(u_grad, lamb, Ks, Kt, N_0, d, dx, dt, M_weight, D_weight)
        return u_lap

    ### Energy Computation of data fidelity function (G), convex conjugate (G*), and regularizer function (F) ###
    def G_denoise(self, u, z, D, M, A):
        res = u - z
        return 1/2*cp.sum(res*(D@(M@res).T).T)
    
    def G_inverse(self, u, z, D, M, A):
        res = A @ u - z
        return 1/(2*A.shape[0])*cp.sum(res*(D@res.T).T)

    def G_denoise_star(self, u, D, A, dual_mat, dual_vec):
        u_in = -u
        return 1/2*cp.sum(u_in*(dual_mat@(D@(u_in).T).T)) + cp.sum(u_in*dual_vec)

    def G_inverse_star(self, u, D, A, dual_mat, dual_vec):
        u_in = -u
        return A.shape[0]/2*cp.sum(u_in*(dual_mat@(D@(u_in).T).T)) + cp.sum(u_in*dual_vec)

    def F_l1(self, p, dx, dt, dx_trap, dt_trap):
        dx_sum = dx.sum(1)
        sum_s = cp.sum(dt_trap[cp.newaxis, cp.newaxis, :] * dx_sum[:, cp.newaxis, cp.newaxis] * cp.abs(p[0]))
        sum_t = cp.sum(dt * dx_trap[:, cp.newaxis] * cp.abs(p[1]))
        return sum_s  + sum_t

    def F_l2(self, p, dx, dt, dx_trap=None, dt_trapt_trap=None):
        return cp.sum(dt/2*dx.T[:,:,cp.newaxis,cp.newaxis] * cp.sqrt(cp.sum(p**2, axis=2)))


    ### Proximal Operators ###
    def prox_G_denoise(self, u, rhs, prox_G_L, tau, M=None):
        return (u + rhs)/(1 + tau)
    
    def prox_G_inverse(self, u, rhs, prox_G_L, tau, M):
        rhs = rhs + M @ u
        return solve_triangular(prox_G_L.T, solve_triangular(prox_G_L, rhs, lower=True), lower=False)

    def prox_F_l1(self, p):
        return cp.clip(p[0], -1, 1), cp.clip(p[1], -1, 1)
    
    def prox_F_l2(self, p):
        return p / cp.maximum(1, cp.linalg.norm(p, axis=2, keepdims=True))


    ### Optimization ###
    def tv_primal_dual(self, u_init, lamb, z, Ks, Kt, A, A_pinv, N_0, d, dx, dt, M, M_weight, D, D_weight, dx_trap, dt_trap, int_op_space, energy): 
        """Total variation regularization - primal dual algorithm"""
        N_1, T = u_init.shape
        
        # Compute Lipschitz constant for step sizes
        if self.norm == 'l1':
            grad = self.grad_l1
            div = self.div_l1
            prox_F = self.prox_F_l1
            
            def grad_space_flat(u):
                assert u.ndim == 1
                return self.grad_space(u.reshape([N_1, T]), lamb[0], Ks, N_0, d).flatten()
                
            def grad_time_flat(u):
                assert u.ndim == 1
                return self.grad_time(u.reshape([N_1, T]), lamb[1], Kt).flatten()
            
            def div_space_flat(p):
                assert p.ndim == 1
                return self.div_space(p.reshape([N_0, d, T]), lamb[0], Ks, dx, N_0, d, M_weight).flatten()
                
            def div_time_flat(p):
                assert p.ndim == 1
                return self.div_time(p.reshape([N_1, T-1]), lamb[1], Kt, D_weight, dt).flatten()
            
            Kst_s = LinearOperator(matvec=grad_space_flat, rmatvec=div_space_flat, dtype=self.dtype, shape=[N_0*d*T, N_1*T])
            Kst_t = LinearOperator(matvec=grad_time_flat, rmatvec=div_time_flat, dtype=self.dtype, shape=[N_1*(T-1), N_1*T])
            
            K_full_s = Kst_s.T@Kst_s
            Kst_eig_s = eigsh(K_full_s, k=1, tol=1e-3)[0][-1]
            K_full_t = Kst_t.T@Kst_t
            Kst_eig_t = eigsh(K_full_t, k=1, tol=1e-3)[0][-1]
            if lamb[1] == 0:
                Lip = cp.sqrt(Kst_eig_s)
            else:
                Lip = cp.sqrt(Kst_eig_s + Kst_eig_t)
            F = self.F_l1
            
        elif self.norm == 'l2':
            grad = self.grad_l2
            div = self.div_l2
            prox_F = self.prox_F_l2
            
            def grad_flat(u):
                assert u.ndim == 1
                return grad(u.reshape([N_1, T]), lamb, Ks, Kt, N_0, d, int_op_space).flatten()
            
            def div_flat(p):
                assert p.ndim == 1
                return div(p.reshape([d, N_0, d+1, 2, T-1]), lamb, Ks, Kt, N_0, d, dx, dt, M_weight, D_weight, int_op_space).flatten()
            
            Kst = LinearOperator(matvec=grad_flat, rmatvec=div_flat, dtype=self.dtype, shape=[d*N_0*(d+1)*2*(T-1), N_1*T])
            K_full = Kst.T@Kst
            Kst_eig = eigsh(K_full, k=1, tol=1e-3)[0][-1]
            Lip = cp.sqrt(Kst_eig)
            
            F = self.F_l2

        u_new = u_old = u = u_init
        p = grad(u, lamb, Ks, Kt, N_0, d, int_op_space)
        
        tau = 1 / Lip
        sigma = 1 / (tau * Lip**2)
        theta = 1
        
        if self.problem == 'denoise':
            prox_G = self.prox_G_denoise
            G = self.G_denoise
            G_star = self.G_denoise_star
            
            rhs = tau * z
            prox_G_L = None
            dual_mat = M
            dual_vec = M@(D@z.T).T
            
        elif self.problem == 'inverse':
            prox_G = self.prox_G_inverse
            G = self.G_inverse
            G_star = self.G_inverse_star
            
            rhs = tau/A.shape[0] * A.T @ z
            prox_G_op = tau/A.shape[0] * A.T @ A + cp.array((M).todense())
            prox_G_L = cholesky(prox_G_op)
            dual_mat = M@A_pinv@A_pinv.T@M
            dual_vec = M@A_pinv@(D@z.T).T
        
        # Primal-dual iterations
        for it in range(self.max_iter):
            u_l = u.copy()
            u_grad = grad(u_new, lamb, Ks, Kt, N_0, d, int_op_space)
            
            if self.norm == 'l1':
                p_in = [p_p + sigma * u_grad_p for p_p, u_grad_p in zip(p, u_grad)]
                p = prox_F(p_in) 
            elif self.norm == 'l2':
                p = prox_F(p + sigma * u_grad)     
                
            p_div = div(p, lamb, Ks, Kt, N_0, d, dx, dt, M_weight, D_weight, int_op_space)
            u = prox_G(u - tau * p_div, rhs, prox_G_L, tau, M)
                        
            u_new = u + theta * (u - u_old)
            u_old = u
            
            # Stopping criterion
            l_inf = cp.max(cp.abs(u_l-u))
            if l_inf < 1e-3:
                return u
            
            if energy and (it+1)%100 == 0:
                current_primal = G(u, z, D, M, A) + F(u_grad, dx, dt, dx_trap, dt_trap)
                current_dual = G_star(p_div, D, A, dual_mat, dual_vec)
                current_gap = current_primal + current_dual
                print(f"Iteration ({it+1}) | Energy {current_primal:.2e} + {current_dual:.2e} = {current_gap:.2e}")
        return u

    def tik_cg(self, u_init, lamb, z, Ks, Kt, A, A_pinv, N_0, d, dx, dt, M, M_weight, D, D_weight, dx_trap, dt_trap, int_op_space, energy):
        """Tikhonov regularization - conjugate gradient algorithm"""
        N_1, T = u_init.shape
        
        if self.norm == 'zero':
            if len(lamb) != 1:
                lamb = lamb[0]
                
            if self.problem == 'denoise':
                u = z / (1 + lamb)
            
            elif self.problem == 'inverse':
                chol_in = A.T @ A + cp.array((M).todense())
                triang = cholesky(chol_in)
                u = solve_triangular(triang.T, solve_triangular(triang, 1/A.shape[0]*A.T @ z, lower=True), lower=False)
            
        elif self.norm == 'first':
            def space_time_laplacian_flat(u):
                assert u.ndim == 1
                return self.space_time_laplacian(u.reshape([N_1, T]), lamb, Ks, Kt, N_0, d, dx, dt, M_weight, D_weight).flatten()
                        
            if self.problem == 'denoise':
                rhs = z
                tik_op = LinearOperator(matvec=lambda x: x + space_time_laplacian_flat(x), shape=[N_1*T, N_1*T], dtype=self.dtype)

            elif self.problem == 'inverse':
                if self.lumped:
                    rhs = 1/A.shape[0] * M_weight * A.T @ z
                    def lhs_flat(u):
                        assert u.ndim == 1
                        return (1/A.shape[0] * M_weight * A.T @ A @ u.reshape([N_1, T])).flatten()
                    
                else:
                    rhs = 1/A.shape[0]*M_weight.solve(A.T @ z)
                    def lhs_flat(u):
                        assert u.ndim == 1
                        return (1/A.shape[0] * M_weight.solve(A.T @ A @ u.reshape([N_1, T]))).flatten()
                    
                tik_op = LinearOperator(matvec=lambda x: lhs_flat(x) + space_time_laplacian_flat(x), shape=[N_1*T, N_1*T], dtype=self.dtype)
                
            res, _ =  cg(tik_op, rhs.flatten(), tol=(1e-10 if self.dtype == np.float64 else 1e-6))
            u = res.reshape([N_1,T])
            
        return u