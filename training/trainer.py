import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from torch.utils import tensorboard
from utils import build_model, normalize, dataset_ecgi, add_Gaussian_noise_dB

from models.optimization import L2, H1

class Trainer:
    """
    """
    def __init__(self, config, device):
        self.config = config

        self.device = device
        if config['precision'] == "double":                
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
        
        self.noise_val = config['noise_val']
        self.noise_range = config['noise_range']
            
        self.valid_epoch_num = 0

        # Dataloaders
        print('Preparing the dataloaders')
        self.batch_size = config["train_dataloader"]["batch_size"] # in general equals one because of different sizes for each "image"

        self.train_dataloader = DataLoader(dataset_ecgi("data/data_csv/train.csv").data_set, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset_ecgi("data/data_csv/val.csv").data_set, batch_size=1, shuffle=True)
        
        # Build the model
        print('Building the model')
        self.model, self.config = build_model(self.config)
        self.model = self.model.to(device, self.dtype)
        print(self.model)
        
        optimizer = torch.optim.Adam

        params_dicts = []
        
        # Experts paramters
        params_dicts.append({"params": self.model.l_op.parameters(), "lr": config["training_options"]["lr_conv"]})
        # Activation parameters
        params_dicts.append({"params": self.model.mu.parameters(), "lr": config["training_options"]["lr_mu"]})
        # Regularization parameters
        params_dicts.append({"params": [self.model.taus, self.model.Q_param, self.model.eps_omega, self.model.lamb], "lr": config["training_options"]["lr_activation"]})

        self.optimizer = optimizer(params_dicts)
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.config['training_options']['lr_decay'])

        # Loss function
        if config["model_params"]["loss"] == "L2":
            self.criterion = L2
        elif config["model_params"]["loss"] == "H1":
            self.criterion = H1
        
        data = np.load("data/data_fixed/fixed_data.npz")
            
        to_tensor = lambda x: torch.from_numpy(x).to(device=self.device, dtype=self.dtype)
        
        self.M           = to_tensor(data['M']).unsqueeze(0).unsqueeze(0)
        if self.model.l_op.lumped:
            M_lumped = self.M.squeeze().sum(dim=0)   
            self.M       = torch.diag(M_lumped).unsqueeze(0).unsqueeze(0)
            self.M_inv   = torch.diag(1/M_lumped).unsqueeze(0).unsqueeze(0)
        else:
            self.M_inv   = to_tensor(data['M_inv']).unsqueeze(0).unsqueeze(0)
        self.dx          = to_tensor(data['dx'])
        self.Ks          = to_tensor(data['Ks']).unsqueeze(0).unsqueeze(0)
        self.A           = to_tensor(data['A']).unsqueeze(0).unsqueeze(0)
        self.proj_p1     = to_tensor(data['proj_p1']).unsqueeze(0).unsqueeze(0)
        self.L_data_fid  = torch.sqrt(to_tensor(data['L_data_fid']))
        
        
        # CHECKPOINTS & TENSOBOARD
        self.checkpoint_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        config_save_path = os.path.join(config['logging_info']['log_dir'], config['exp_name'], f'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(getattr(self, f'config'),handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)


    def train(self):
        self.batch_seen = 0
        while self.batch_seen < self.config["training_options"]["n_batches"]:
            self.train_epoch()

        self.writer.flush()
        self.writer.close()

        
    def train_epoch(self):
        """
        """
        self.model.train()
        tbar = tqdm(self.train_dataloader, ncols=80, position=0, leave=True)
        log = {}
        for batch_idx, sample in enumerate(tbar):
            self.batch_seen += 1
            
            # Validation and saving checkpoitns
            if (self.batch_seen % self.config["logging_info"]["log_batch"]-1) == 0:
                self.valid_epoch()
                self.model.train()
                self.save_checkpoint(self.batch_seen)
            
            # Scheduler step
            if self.batch_seen % self.config['training_options']['n_batch_decay'] == 0:
                self.scheduler.step()
            
            # Load batch of data functions and the timestep size
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
            
            # Generate temporal gradient matrix
            Kt = - torch.eye(t, dtype=self.dtype, device=self.device)
            Kt[:-1, 1:] += torch.eye(t-1, dtype=self.dtype, device=self.device)
            Kt = 1/dt * Kt[:-1].unsqueeze(0).unsqueeze(0)
            
            # stopping criterion
            if self.batch_seen > self.config["training_options"]["n_batches"]:
                break
            
            sigma = torch.torch.empty((data.shape[0], 1, 1, 1), device=data.device, dtype=torch.float64).uniform_(self.noise_range[0], self.noise_range[1])
            
            # Generate noise observations
            if self.config['model_params']['problem'] == "denoise":
                noise = sigma * torch.randn(data.shape,device=data.device, dtype=self.dtype)
                noisy_data = data + noise
            elif self.config['model_params']['problem'] == "inverse":
                noisy_data = add_Gaussian_noise_dB(self.A @ data, 1/sigma)
            
            self.optimizer.zero_grad()
            output = self.model(noisy_data, sigma, self.proj_p1, self.Ks, self.M, self.M_inv, D, D_inv, dt, self.A, self.L_data_fid)
            
            loss = (self.criterion(output/sigma.sqrt(), data/sigma.sqrt(), self.M, D, self.Ks, Kt, self.dx, dt))
            loss.backward()
            self.optimizer.step()
     

            log['loss'] = loss.item()
            log['sigma'] = sigma.mean(0).item()
            log['forward_mean_iter'] = self.model.fw_niter_mean
            log['forward_max_iter'] = self.model.fw_niter_max
            log['power_iterations'] = self.model.l_op.power_iteration
            log['foe_lipschitz'] = self.model.l_op.L
            
            log['lamb'] = self.model.lamb.exp()
            log['eps_omega'] = self.model.eps_omega.exp()
            log['eps_theta'] = self.model.l_op.eps_theta.exp()
            
            self.wrt_step = self.batch_seen
            self.write_scalars_tb(log)
            tbar.set_description(f"T ({self.valid_epoch_num}) | TotalLoss {log['loss']:.7f}")

        return log


    def valid_epoch(self):
        self.valid_epoch_num += 1

        loss_val = 0.0
        val_size = 0
        tbar_val = tqdm(self.val_dataloader, ncols=80, position=0, leave=True)
        with torch.no_grad():
            for batch_idx, sample in enumerate(tbar_val):
                # Load batch of data functions and the timestep size
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
                
                # Generate temporal gradient matrix
                Kt = - torch.eye(t, dtype=self.dtype, device=self.device)
                Kt[:-1, 1:] += torch.eye(t-1, dtype=self.dtype, device=self.device)
                Kt = 1/dt * Kt[:-1].unsqueeze(0).unsqueeze(0)

                # Generate noise observations
                sigma = self.noise_val * torch.ones((data.shape[0], 1, 1, 1), device=data.device, dtype=self.dtype)
                if self.config['model_params']['problem'] == "denoise":
                    noise = sigma * torch.randn(data.shape,device=data.device, dtype=self.dtype)
                    noisy_data = data + noise
                elif self.config['model_params']['problem'] == "inverse":
                    noisy_data = add_Gaussian_noise_dB(self.A @ data, 1/sigma)
                
                output = self.model(noisy_data, sigma, self.proj_p1, self.Ks, self.M, self.M_inv, D, D_inv, dt, self.A, self.L_data_fid)
                loss = (self.criterion(output, data, self.M, D, self.Ks, Kt, self.dx, dt))
                
                loss_val += loss.cpu().item()

                data.detach()
                output.detach()
                loss.detach()
                        
                val_size += 1                
                    
        loss_val = loss_val/len(self.val_dataloader)
        tbar_val.set_description('EVAL ({}) | L2Loss: {:.5f}'.format(self.valid_epoch_num, loss_val))
        
        self.wrt_mode = 'Convolutional'
        self.writer.add_scalar(f'{self.wrt_mode}/Validation loss', loss_val, self.valid_epoch_num )
        log = {'val_loss': loss_val}
        
        # Plot filter dofs
        img = normalize(self.model.l_op.get_filters(self.proj_p1, self.Ks, dt, d=2).detach().cpu())
            
        self.writer.add_image('params', img, self.valid_epoch_num)

        return log

    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'Convolutional/Training {k}', v, self.wrt_step)

    def save_checkpoint(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict()
        }

        print('Saving a checkpoint:')
        # Checkpoints & tensorboard
        self.checkpoint_dir = os.path.join(self.config["logging_info"]['log_dir'], self.config["exp_name"], 'checkpoints')

        filename = self.checkpoint_dir + '/checkpoint_' + str(epoch) + '.pth'
        torch.save(state, filename)
