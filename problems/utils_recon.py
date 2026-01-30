import math
import numpy as np
import torch
import cupy as cp
from cupyx.scipy.sparse import diags, eye, coo_matrix, csr_matrix
from cupyx.scipy.sparse.linalg import splu
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import dataset_ecgi, load_model, add_Gaussian_noise_dB, add_Gaussian_noise_dB_cp
from models.optimization import AGDR, L2, H1
from models.base_methods import Base_Methods, L2_cp, H1_cp

torch.set_grad_enabled(False)


class Tune_Hyperparams_MFoE:
    """Tune hyperparameters for the MFoE method"""
    def __init__(self, config, device):
        self.config = config
        
        self.device = device
        if config['method']['precision'] == "double":                
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
        
        self.noise_val = config['noise_val']
        self.max_iter = config['method']['max_iter']
        
        self.test_dataloader = DataLoader(dataset_ecgi("data/data_csv/test.csv").data_set, batch_size=1)
        self.val_dataloader = DataLoader(dataset_ecgi("data/data_csv/val.csv").data_set, batch_size=1)
        
        self.model = load_model(self.config['method']['reg'], device, reg = config['method']['problem'])
        self.model = self.model.to(device, self.dtype)
        
        if config['method']['loss'] == "L2":
            self.criterion = L2
        elif config['method']['loss'] == "H1":
            self.criterion = H1
        
        data = np.load("data/data_fixed/fixed_data.npz")
            
        to_tensor = lambda x: torch.from_numpy(x).to(device=self.device, dtype=self.dtype)
        
        self.M           = to_tensor(data['M']).unsqueeze(0).unsqueeze(0)
        if self.model.l_op.lumped:
            M_lumped     = self.M.squeeze().sum(dim=0)   
            self.M       = torch.diag(M_lumped).unsqueeze(0).unsqueeze(0)
            self.M_inv   = torch.diag(1/M_lumped).unsqueeze(0).unsqueeze(0)
        else:
            self.M_inv   = to_tensor(data['M_inv']).unsqueeze(0).unsqueeze(0)
        self.dx          = to_tensor(data['dx'])
        self.Ks          = to_tensor(data['Ks']).unsqueeze(0).unsqueeze(0)
        self.A           = to_tensor(data['A']).unsqueeze(0).unsqueeze(0)
        self.proj_p1     = to_tensor(data['proj_p1']).unsqueeze(0).unsqueeze(0)
        self.L_data_fid  = torch.sqrt(to_tensor(data['L_data_fid']))
                
    def tune(self, reg_param_in):
        """Tune parameters on the validation set"""
        lamb, sigma = reg_param_in
        self.model.lamb.data = self.model.lamb + math.log(lamb)
        
        loss_test = 0.0
        tbar = tqdm(self.val_dataloader, desc="Tuning", ncols=80)
        for batch_idx, sample in enumerate(tbar):
            data = sample[0].unsqueeze(0).to(self.device, self.dtype) 
            dt = sample[1].to(self.device, self.dtype)    
                
            t = data.shape[-1]
            if self.model.l_op.lumped:
                    D = dt * torch.eye(t).unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)
                    D_inv = 1/dt * torch.eye(t).unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)
            else:
                main_diag = torch.full((t,), 2/3)
                main_diag[0] = main_diag[-1] = 1/3
                off_diag = torch.full((t-1,), 1/6)
                D = dt * (torch.diag(main_diag) + torch.diag(off_diag, diagonal=1) + torch.diag(off_diag, diagonal=-1)).unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)
                D_inv = torch.linalg.inv(D)
            Kt = - torch.eye(t, dtype=self.dtype, device=self.device)
            Kt[:-1, 1:] += torch.eye(t-1, dtype=self.dtype, device=self.device)
            Kt = Kt[:-1].unsqueeze(0).unsqueeze(0)
            
            noise_vec = self.noise_val * torch.ones((data.shape[0], 1, 1, 1), device=data.device, dtype=self.dtype)
            if self.config['method']['problem'] == "denoise":
                noise = noise_vec * torch.randn(data.shape,device=data.device, dtype=self.dtype)
                noisy_data = data + noise
                data_init = noisy_data.clone()
            elif self.config['method']['problem'] == "inverse":
                noisy_data = add_Gaussian_noise_dB(self.A @ data, 1/noise_vec)
                data_init = torch.zeros_like(data)
                
            _, _, space_nodes, time_steps = data.shape
            if not self.model.l_op.approx:
                self.model.l_op.precompute_fem_matrix(time_steps, noisy_data.device, noisy_data.dtype)
            self.model.l_op.spectral_norm(space_nodes, time_steps, self.proj_p1, self.Ks, self.M, self.M_inv, D, D_inv, dt, self.model.d)
            
            output = AGDR(data_init, noisy_data, self.proj_p1, self.Ks, self.M, self.M_inv, D, D_inv, dt, self.A, self.L_data_fid, self.model, sigma * torch.ones(1, 1, 1, 1, device=self.device, dtype=self.dtype), self.max_iter)[0]
            loss = (self.criterion(output, data, self.M, D, self.Ks, Kt, self.dx))
            loss_test += loss.cpu().item()       
                                  
        loss_test_mean = loss_test/len(self.val_dataloader)
        self.model.lamb.data = self.model.lamb - math.log(lamb)  
        
        return loss_test_mean
    
    def apply(self, reg_param_in):
        """Evalution of the parameters on the test set"""
        lamb, sigma = reg_param_in
        self.model.lamb.data = self.model.lamb + math.log(lamb)
        
        loss_test = 0.0
        output_list = []
        loss_list = []
        data_list = []
        dt_list = []
        tbar = tqdm(self.test_dataloader, desc="Testing", ncols=80)
        for batch_idx, sample in enumerate(tbar):
            data = sample[0].unsqueeze(0).to(self.device, self.dtype)  
            dt = sample[1].to(self.device, self.dtype)   
                
            t = data.shape[-1]
            if self.model.l_op.lumped:
                    D = dt * torch.eye(t).unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)
                    D_inv = 1/dt * torch.eye(t).unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)
            else:
                main_diag = torch.full((t,), 2/3)
                main_diag[0] = main_diag[-1] = 1/3
                off_diag = torch.full((t-1,), 1/6)
                D = dt * (torch.diag(main_diag) + torch.diag(off_diag, diagonal=1) + torch.diag(off_diag, diagonal=-1)).unsqueeze(0).unsqueeze(0).to(self.device, self.dtype)
                D_inv = torch.linalg.inv(D)
            Kt = - torch.eye(t, dtype=self.dtype, device=self.device)
            Kt[:-1, 1:] += torch.eye(t-1, dtype=self.dtype, device=self.device)
            Kt = Kt[:-1].unsqueeze(0).unsqueeze(0)
            
            noise_vec = self.noise_val * torch.ones((data.shape[0], 1, 1, 1), device=data.device, dtype=self.dtype)
            if self.config['method']['problem'] == "denoise":
                noise = noise_vec * torch.randn(data.shape,device=data.device, dtype=self.dtype)
                noisy_data = data + noise
                data_init = noisy_data.clone()
            elif self.config['method']['problem'] == "inverse":
                noisy_data = add_Gaussian_noise_dB(self.A @ data, 1/noise_vec)
                data_init = torch.zeros_like(data)
                
            _, _, space_nodes, time_steps = data.shape
            if not self.model.l_op.approx:
                self.model.l_op.precompute_fem_matrix(time_steps, noisy_data.device, noisy_data.dtype)
            self.model.l_op.spectral_norm(space_nodes, time_steps, self.proj_p1, self.Ks, self.M, self.M_inv, D, D_inv, dt, self.model.d)

            output = AGDR(data_init, noisy_data, self.proj_p1, self.Ks, self.M, self.M_inv, D, D_inv, dt, self.A, self.L_data_fid, self.model, sigma * torch.ones(1, 1, 1, 1, device=self.device, dtype=self.dtype), self.max_iter)[0]
            output_list.append(output.squeeze().cpu().numpy())
            if self.config['save_gt_data']:
                data_list.append(data.squeeze().cpu().numpy())
                dt_list.append(dt.item())
            loss = (self.criterion(output, data, self.M, D, self.Ks, Kt, self.dx, dt)) 
            loss_test += loss.cpu().item()    
            loss_list.append(loss.cpu().item())   
                                  
        loss_list = np.array(loss_list)
        loss_test_mean = loss_test/len(self.test_dataloader)
        self.model.lamb.data = self.model.lamb - math.log(lamb)  
        
        return loss_test_mean, loss_list, output_list, data_list, dt_list
    

class Tune_Hyperparams_Base:
    """Tune hyperparameters for the baseline methods"""
    def __init__(self, config, device):
        self.config = config

        self.device = device
        if config['method']['precision'] == "double":                
            self.dtype = cp.float64
        else:
            self.dtype = cp.float32
        
        self.noise_val = config['noise_val']
        
        self.test_dataloader = DataLoader(dataset_ecgi("data/data_csv/test.csv").data_set, batch_size=1)
        self.val_dataloader = DataLoader(dataset_ecgi("data/data_csv/val.csv").data_set, batch_size=1)
        
        self.model = Base_Methods(config['method'], device)
        
        if config['method']['loss'] == "L2":
            self.criterion = L2_cp
        elif config['method']['loss'] == "H1":
            self.criterion = H1_cp
            
        data = np.load("data/data_fixed/fixed_data.npz")
        data_interpol = np.load("data/data_fixed/fixed_data_base.npz")
        to_cp = lambda x: cp.array(x).astype(self.dtype)
        to_cp_csr = lambda x: csr_matrix(cp.asarray(x, dtype=self.dtype))

        self.M            = to_cp_csr(np.array(data['M']))
        if self.model.lumped:
            M_diag        = self.M.sum(1)
            self.M        = diags(cp.asarray(M_diag).flatten())
            self.M_weight = cp.array(1/(cp.asarray(M_diag).flatten()))[:,cp.newaxis]
            self.M        = to_cp_csr(np.array(data['M']))
        else:
            self.M_weight = splu(self.M)
        
        self.dx               = to_cp(data['dx'])
        self.Ks               = to_cp_csr(data['Ks'])
        self.A                = to_cp(data['A'])
        self.A_pinv           = cp.linalg.pinv(self.A)
        self.int_op_space     = to_cp_csr(data_interpol['int_op_space'])
        self.proj_elem_to_dof = to_cp_csr(data_interpol['proj_elem_to_dof'])
        
        self.d = self.dx.shape[-1]
        self.N_0 = self.dx.shape[0]
        self.N_1 = self.Ks.shape[-1]
        self.nb_elec = self.A.shape[0]
        self.dx_trap = self.proj_elem_to_dof@(self.dx.sum(1))
        
        
    def tune(self, lamb):
        """Tune parameters on the validation set"""
        loss_test = 0.0
        tbar = tqdm(self.val_dataloader, desc="Tuning", ncols=80)
        for batch_idx, sample in enumerate(tbar):
            data = cp.array(sample[0].squeeze(0).numpy(), dtype=self.dtype)

            t = data.shape[-1]
            dt =  sample[1].item()
            main_diag = cp.full(t, 2/3)
            main_diag[0] = main_diag[-1] = 1/3
            off_diag = cp.full(t-1, 1/6) 
            D = 1/dt* diags(
                diagonals=[main_diag, off_diag, off_diag],
                offsets=[0, -1, 1],
                format='csr',
                dtype = self.dtype)
            dt_trap = cp.asarray(D.sum(1)).flatten()
            if self.model.lumped:
                D = diags(dt_trap)
                D_weight = cp.array(1/dt_trap)[cp.newaxis]
            else:
                D_weight = splu(D)
            Kt = 1/dt * (-eye(t, dtype=self.dtype) + eye(t, k=1, dtype=self.dtype))
            Kt = coo_matrix(Kt.todense()[:-1])

            if self.config['method']['problem'] == "denoise":
                noise = self.noise_val * cp.random.randn(*data.shape, dtype=self.dtype)
                noisy_data = data + noise
            elif self.config['method']['problem'] == "inverse":
                noisy_data = add_Gaussian_noise_dB_cp(self.A @ data, 1/self.noise_val)
        
            data_init = cp.zeros_like(data)
            output = self.model.optim(data_init, lamb, noisy_data, self.Ks, Kt, self.A, self.A_pinv , self.N_0, self.d, self.dx, dt, self.M, self.M_weight, D, D_weight, self.dx_trap, dt_trap, self.int_op_space, energy=False)    

            loss = (self.criterion(output, data, self.M, D, self.Ks, Kt, self.dx, dt))
            loss_test += loss.item()       
                                  
        loss_test_mean = loss_test/len(self.val_dataloader)
        
        return loss_test_mean
    
    def apply(self, lamb):
        lamb = (0.00354,0.00324)
        """Evalution of the parameters on the test set"""
        to_np = lambda x: (x.get() if issubclass(type(x), cp.ndarray) else x)
        
        loss_val = 0.0
        output_list = []
        data_list = []
        dt_list = []
        loss_list = []
        tbar = tqdm(self.test_dataloader, desc="Testing", ncols=80)
        for batch_idx, sample in enumerate(tbar):
            data = cp.array(sample[0].squeeze(0).numpy(), dtype=self.dtype)
            
            t = data.shape[-1]
            dt =  sample[1].item()
            main_diag = cp.full(t, 2/3)
            main_diag[0] = main_diag[-1] = 1/3
            off_diag = cp.full(t-1, 1/6) 

            D = 1/dt* diags(
                diagonals=[main_diag, off_diag, off_diag],
                offsets=[0, -1, 1],
                format='csr',
                dtype = self.dtype)
            
            dt_trap = cp.asarray(D.sum(1)).flatten()
            
            if self.model.lumped:
                D = diags(dt_trap)
                D_weight = cp.array(1/dt_trap)[cp.newaxis]
            else:
                D_weight = splu(D)
            
            Kt = 1/dt * (-eye(t, dtype=self.dtype) + eye(t, k=1, dtype=self.dtype))
            Kt = coo_matrix(Kt.todense()[:-1])

            if self.config['method']['problem'] == "denoise":
                noise = self.noise_val * cp.random.randn(*data.shape)
                noisy_data = data + noise
            elif self.config['method']['problem'] == "inverse":
                noisy_data = add_Gaussian_noise_dB_cp(self.A @ data, 1/self.noise_val)
        
            data_init = cp.zeros_like(data)
            output = self.model.optim(data_init, lamb, noisy_data, self.Ks, Kt, self.A, self.A_pinv , self.N_0, self.d, self.dx, dt, self.M, self.M_weight, D, D_weight, self.dx_trap, dt_trap, self.int_op_space, energy=False)    
            output_list.append(to_np(output))
            if self.config['save_gt_data']:
                data_list.append(to_np(data))
                dt_list.append(dt)
            
            loss = (self.criterion(output, data, self.M, D, self.Ks, Kt, self.dx, dt))
            loss_val += loss.item()    
            loss_list.append(loss.item())   
                                  
        loss_list = np.array(loss_list)
        loss_val_mean = loss_val/len(self.test_dataloader)
        
        return loss_val_mean, loss_list, output_list, data_list, dt_list



