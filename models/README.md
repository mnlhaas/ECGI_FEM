# Models and Optimization

This directory contains the PyTorch implementations of the **Temporal Multivariate Fields-of-Experts (MFoE)** model, the associated learned linear operators, and the optimization routines used for solving the inverse problem of Electrocardiographic Imaging (ECGI).

## File Structure

### 1. Deep Learning Architecture

*   **`mfoe.py`**: Contains the main model class `MFoE_temp`. This defines the variational network architecture which unrolls an optimization algorithm (like Gradient Descent) where the regularization functional is learned.
*   **`l_operator.py`**: Implements the learned linear operators ($L_i$) used within the MFoE regularization term.
    *   **`FemConvolution`**: A custom layer that performs temporal convolution combined with spatial FEM operations.
    *   **`L_Operator`**: Aggregates spatial gradients (via Stiffness matrix $K_s$), temporal convolutions, and identity terms to form the feature maps. It includes spectral normalization to ensure stability.

### 2. Optimization & Loss Functions

*   **`optimization.py`**: Provides the solvers and metrics for the reconstruction process.
    *   **`AGDR`**: Accelerated Gradient Descent Reconstruction (FISTA) used to minimize the variational cost function.
    *   **Loss Functions**: `L2`, `L2_squared`, and `H1` norms defined over the mesh using FEM matrices (Spatial mass matrix $M$, temporal mass matrix $D$, spatial gradient $K_s$, temporal gradient $K_t$).
    *   **`proj_l1_channel`**: Projection operator onto the $L_1$-ball, used in proximal gradient steps.

### 3. Baseline Methods

*   **`methods.py`**: Contains GPU-accelerated (CuPy) implementations of classical regularization methods for comparison:
    *   Tikhonov (`TIK`) Regularization (Zero-order  (`zero`) and First-order (`first`)).
    *   Total Variation  (`TV`) (Isotropic (`l2`) and Anisotropic  (`l1`)).


`lumped` acclerates the optimization of both methods by using row-sum lumping of the FEM matrices (including the convolution matrix). `approx` simplifies the network by neglecting boundary conditions in the convolution.


