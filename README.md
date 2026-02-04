# LEARNED FEM REGULARIZATION FOR ECGI

The repository contains the implementation of a **Finite Element-based Multivariate Fields-of-Experts** model designed for solving space-time inverse problems, with a specific application to **Electrocardiographic Imaging (ECGI)** on a 2D torso-heart-model with lungs.
<p align="center">
  <img src="data_generation/figures/torso.pdf" alt="Torso-heart-model with lungs and electrodes" width="400"/>
</p>

The project includes a complete pipeline for:
1.  Generating synthetic cardiac electrophysiology data using Finite Element Methods (FEM).
2.  Training a deep learning model (`MFoE_temp`) to reconstruct cardiac potentials.
3.  Comparing the learned model against classical spatiotemporal baseline methods (Tikhonov Regularization, Total Variation).

## Features

*   **Synthetic Data Generation**: Simulates cardiac electrical activity using a monodomain model with Nagumo ionic dynamics on 2D meshes.
*   **Deep Learning Model**: Implements a space-time multivariateField of Experts model based on [Ducotterd et al. (2025)](https://arxiv.org/pdf/2508.06490?) using PyTorch, incorporating FEM operators (Mass, Gradient) directly into the network architecture.
*   **Baseline Methods**: Includes GPU-accelerated implementations (via CuPy) of classical regularization techniques:
    *   (`zero`)- and (`first`)-order Tikhonov (`TIK`) Regularization solved using the conjugate gradient method.
    *   Anisotropic (`l1`) and isotropic (`l2`) Total Variation (`TV`) using Primal-Dual Hybrid Gradient algorithms. The space-time TV implementation is based on the work of [Haas et al. (2025)](https://doi.org/10.1137/24M1685055).
*   **Inverse & Denoising**: Supports both direct denoising of mesh functions and solving the ill-posed inverse problem from sparse torso electrode measurements.

## Requirements

*   Python 3.8+
*   CUDA-capable GPU (recommended for training and CuPy baselines)

### Dependencies

Install the required Python packages by the enviroment file `environment.yml` or by running the following commands:

```bash
pip install torch numpy scipy pandas matplotlib tqdm tensorboard
pip install cupy-cuda11x  # Adjust based on your CUDA version
pip install scikit-fem meshio pyvista torchdeq scikit-sparse
```

## Usage

### 1. Data Generation

Before training, you must generate the synthetic dataset and the fixed FEM operators.

```bash
python data_generation/gen_data.py
```

This script performs the following:
*   Generates synthetic cardiac potential samples based on `data_generation/config_data.json` including scar tissue.
*   Computes the forward problem to map potentials to torso electrodes.
*   Saves the dataset to `data/data_functions/`.
*   Precomputes FEM matrices (Mass, Stiffness, Projection) and saves them to `data/data_fixed/`.
*   Splits data into train/test/val CSV files.

### 2. Training

To train the MFoE model run:

```bash
python training/train.py --device cpu or cuda:n
```

**Key Configuration Options (`training/config_train.json`):**
*   `logging_info`:     Directory paths for logs and checkpoints.
*   `model_params`:     Specifies the problem type ("inverse" or "denoise"), the loss function ("L2" or "H1"), and the model architecture.
*   `optimization`:     Forward and backward pass parameters.
*   `training_options`: Learning rates and number of iterations.

Training progress, including loss curves and parameter visualizations, can be monitored using TensorBoard:

```bash
tensorboard --logdir logs/
```

### 3. Reconstruction & Evaluation

To evaluate the trained model or run baseline reconstruction methods:

```bash
python problems/reconstruct.py --device cuda:n (or cpu for MFoE)
```

**Key Configuration Options (`problems/config_recon.json`):**
*   `logging_info`: Directory paths for logs.
*   `regularizer`:  Choose between `"MFoE"` (learned model) or `"base"` (classical methods).
*   `method`:       The `problem` parameter can be chosen either `"denoise"` or `"inverse"` for the inveres problem of ECGI. `reg` denotes either the base method `TIK` or `TV` or the learned model name. The `norm` parameter determines the specific type of regularization (`zero` or `first`/`l1` or `l2`).
*   `tune`:         Whether to perform hyperparameter tuning (finding optimal $\lambda$, $\sigma$).


## Structure

```text
temporal_foe/
├── data/                   # Dataset and fixed FEM operators
├── data_generation/        # Synthetic data creation
│   ├── gen_data.py         # Main generation script
│   ├── utils_data.py       # FEM and mesh utilities
│   └── config_data.json    # Simulation parameters
├── models/                 # PyTorch model definitions
│   ├── base_methods.py     # Baseline methods (Tikhonov, TV) in CuPy
│   ├── mfoe.py             # Main multivariate FoE temporal model
│   ├── l_operator.py       # Learned operator layers
|   └── optimization.py     # Acclerated gradient descent algorithm
├── problems/               # PyTorch model definitions
│   ├── config_recon.json   # Reconstruction parameters
│   ├── plot.ipynb          # Visualization notebook
│   ├── reconstruct.py      # Inference and comparison script
│   └── utils_recon.py      # Reconstruction utilities
├── trained_models/         # Checkpoints of trained models
├── training/               # Training of MFoE model
│   ├── config.json         # Training configuration
│   ├── train.py            # Training entry point
│   └── trainer.py          # Training loop and validation
└── utils.py                # General utilities (loading, noise, dataset)
```
