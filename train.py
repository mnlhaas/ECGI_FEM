import numpy as np
import torch
import random
import os
import json
import argparse
from training.trainer import Trainer


def main(device):
    # Set up directories for saving results
    config_path = 'training/config_train.json'
    config = json.load(open(config_path))

    exp_dir = os.path.join(config['logging_info']['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    trainer_inst = Trainer(config, device)
   
    trainer_inst.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        help="Device (e.g., cpu or cuda:n)"
    )

    args = parser.parse_args()

    main(args.device)
