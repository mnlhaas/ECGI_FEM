import json
import os
import glob
import torch
import numpy as np
import cupy as cp
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from models.mfoe import MFoE_temp

def add_Gaussian_noise_dB(u, db):
    """ Generate noisy observations with SNR in dB in PyTorch."""
    avg_db_in = 10 ** (db.squeeze(1).squeeze(1).squeeze(1).item() / 10)
    u_var = torch.mean(u**2)

    noise_std = torch.sqrt(u_var/avg_db_in)
    noise = torch.randn_like(u)

    noisy = u + noise_std*noise
    return noisy

def add_Gaussian_noise_dB_cp(u, db):
    """ Generate noisy observations with SNR in dB in CuPy."""
    avg_db_in = 10 ** (db / 10)
    u_var = cp.mean(u**2)

    noise_std = cp.sqrt(u_var/avg_db_in)
    noise = cp.random.randn(*u.shape)

    noisy = u + noise_std*noise
    return noisy

def load_model(name, device='cuda:0', epoch=None, reg=None):
    """ Load model from a checkpoint.
    (Copyright (c) 2025 Stanislas Ducotterd) """
    script_path = Path(__file__).resolve()  
    project_root = script_path.parent  #
    trained_models_path = project_root / "trained_models"
    
    directory = f'{trained_models_path}/{name}/'
    directory_checkpoints = f'{directory}checkpoints/'

    # retrieve last checkpoint
    if epoch is None:
        files = glob.glob(f'{directory}/checkpoints/*.pth', recursive=False)
        epochs = map(lambda x: int(x.split("/")[-1].split('.pth')[0].split('_')[1]), files)
        epoch = max(epochs)
        print("Epoch:",epoch)

    checkpoint_path = f'{directory_checkpoints}checkpoint_{epoch}.pth'
    
    # config file
    config = json.load(open(f'{directory}config.json'.replace("[[]","[").replace("[]]","]")))
   
    config['model_params']['problem'] = reg
   
    # build model
    model, _ = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    return model

def build_model(config):
    model = MFoE_temp(model_params=config['model_params'], l_op_params=config['l_op_params'],
                          fw_param=config['optimization']['fixed_point_solver_fw_params'],
                          bw_param=config['optimization']['fixed_point_solver_bw_params'])
    
    if config['precision'] == 'float':
        model = model.float()
    elif config['precision'] == 'double':
        model = model.double()
    
    return model, config

def normalize(func):
    return (func - func.min())/ (func.max() - func.min())

class dataset_ecgi(Dataset):
    """Mesh function denoising data set."""
    def __init__(self, csv_dir):
        """
        Args:
            csv_dir (string): Path to the csv data set descirption file.
        """

        data_set_csv = pd.read_csv(csv_dir)
        data_set_np = data_set_csv.to_numpy().squeeze()
        
        self.root_dir = "data/data_functions/"

        self.data_set = []
        for i in tqdm(range(len(data_set_np)), ncols=80):
            self.data_set.append(self.load_all(data_set_np[i]))
            
    def load_all(self, name):
        # Load all Information
        func_name = os.path.join(self.root_dir, name)
        data = np.load(func_name)
        func_target = torch.from_numpy(normalize(data["u"]))
        return [func_target, data["dt"]]

    def __len__(self):
        return len(self.data_set) 
    

