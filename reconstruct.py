import os
import numpy as np
import cupy as cp
import random
import json
import torch
import argparse
import sys
import itertools

from problems.utils_recon import Tune_Hyperparams_MFoE, Tune_Hyperparams_Base


def main(device):
    config_path = 'problems/config_recon.json'
    config = json.load(open(config_path))

    exp_dir = os.path.join(config['logging_info']['log_dir'], config['logging_info']['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    if config['log']:
        log_file = os.path.join(exp_dir, f"output.log")
        sys.stdout = open(log_file, "w")
        sys.stderr = sys.stdout
    
    save_path = os.path.join(exp_dir, 'config_recon.json')
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    seed = 42
    torch.set_num_threads(10)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print(config)
    
    best_sigma = config['params']['sigma']
    best_lamb = config['params']['lambda']
    
    def local_grid_search(
        hyper,
        best_params, 
        gamma, 
        print_fn, 
        log_search=False,
        log_scales=3
    ):
        loss_errors = {}
        best_loss = float("inf")

        # Coarse log search
        if log_search:
            grids = [p * np.logspace(-5, 0, log_scales) for p in best_params]
            for params in itertools.product(*grids):
                loss = hyper.tune(params)
                loss_errors[params] = loss
                print_fn(params, loss)

            best_params = min(loss_errors, key=loss_errors.get)
            best_loss = loss_errors[best_params]

        # Local refinement
        while gamma > 1.05:
            grids = [[p/gamma, p, p*gamma] for p in best_params]

            for params in itertools.product(*grids):
                if loss_errors:
                    prev = np.array(list(loss_errors.keys()))
                    rel = np.abs(prev - params) / params
                    if rel.sum(axis=1).min() < 1e-2:
                        continue

                loss = hyper.tune(params)
                loss_errors[params] = loss
                print_fn(params, loss)

            current_best = min(loss_errors.values())
            if current_best < best_loss - 1e-3:
                best_loss = current_best
                best_params = min(loss_errors, key=loss_errors.get)
            else:
                gamma = np.sqrt(gamma)
                print("New gamma:", gamma)

        return best_params
    
    if config['regularizer'] == 'MFoE':
        hyper = Tune_Hyperparams_MFoE(config, device)

        if config['tune']:
            gamma = float(config['params']['gamma'])

            best_lamb, best_sigma = local_grid_search(
                hyper,
                best_params=(best_lamb, best_sigma),
                gamma=gamma,
                log_search=config['params']['log_search_coarse'],
                print_fn=lambda p, loss: print(f"Lambda: {p[0]:.5e}, Sigma: {p[1]:.5e}, loss: {loss:.5e}")
            )
        print(f'Best lambda: {best_lamb:.5e}, Best Sigma: {best_sigma:.5e}')
        best_params=(best_lamb, best_sigma)
        loss_val_mean, loss_list, output_list, data_list, dt_list = hyper.apply(best_params)
        print(f'Validation loss: {loss_val_mean:.5e}')
    
    elif config['regularizer'] == 'base':
        hyper = Tune_Hyperparams_Base(config, device)

        best_lamb_g = best_lamb_t = best_lamb

        if config['tune']:
            gamma = config['params']['gamma']

            best_lamb_g, best_lamb_t = local_grid_search(
                hyper,
                best_params=(best_lamb_g, best_lamb_t),
                gamma=gamma,
                log_search=config['params']['log_search_coarse'],
                print_fn=lambda p, loss: print(f"Lambda_gamma: {p[0]:.5e}, Lambda_t: {p[1]:.5e}, loss: {loss:.5e}")
            )
        best_params = (best_lamb_g, best_lamb_t)
        print(f'Best Lambda_gamma: {best_lamb_g:.5e}, Lambda_t: {best_lamb_t:.5e}')
        loss_val_mean, loss_list, output_list, data_list, dt_list = hyper.apply(best_params)
        print(f'Validation loss: {loss_val_mean:.5e}')
        
    np.save(os.path.join(exp_dir, 'loss_list'), loss_list)
    np.savez(os.path.join(exp_dir, "output_list.npz"), *output_list)
    if config['save_gt_data']:
        val_dir = os.path.join(config['logging_info']['log_dir'], "val_data")
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)    
        np.savez(os.path.join(val_dir, "output_list.npz"), *data_list)
        np.save(os.path.join(val_dir, "dt_list.npy"), np.array(dt_list))
        
    


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')

    args = parser.parse_args()

    main(args.device)



