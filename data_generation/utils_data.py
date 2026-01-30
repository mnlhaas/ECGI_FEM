import numpy as np
from scipy.sparse import coo_matrix, vstack
import pyvista as pv
import meshio

from itertools import product
import multiprocessing
import matplotlib.pyplot as plt

from skfem import Mesh, ElementTetP1, ElementTriP1, Basis, DiscreteField, Element, penalize, condense
from skfem.io import from_meshio
from skfem import BilinearForm
from skfem.helpers import dot, grad
from sksparse.cholmod import cholesky


def normalize_points_minmax(pts):
    pts = np.asarray(pts)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return ((pts - mins) / (maxs - mins)).T

def convert_mesh(mesh : pv.UnstructuredGrid):
    tris = mesh.cells_dict[pv.cell.CellType.TRIANGLE]
    return from_meshio(meshio.Mesh(mesh.points[..., :2], {"triangle": tris}))

def convert_mesh_pts_tris(pts, tris):
    return from_meshio(meshio.Mesh(pts, {"triangle": tris}))

def angle_between(v1, v2=None):
    ang1 = np.arctan2(*(v1.T))
    if v2 is None:
        ang2 = 0.
    else:
        ang2 = np.arctan2(*(v2.T))

    return (ang1 - ang2) % (2 * np.pi)

@BilinearForm
def mass(u, v, _):
    """Assemble spatial mass matrix."""
    return u * v

@BilinearForm(nthreads=multiprocessing.cpu_count() // 2)
def laplace(u, v, w):
    """Assemble spatial Laplace operator."""
    if "sigma00" in w:
        d = w["d"]
        sigma = np.stack([w[f"sigma{i}{j}"] for i, j in product(range(d), range(d))], axis=0)
        sigma = sigma.reshape([d, d, u.shape[0], u.shape[1]])
        return np.einsum('x...,xy...,y...->...', grad(u), sigma, grad(v))
    else:
        return dot(grad(u), grad(v))
    
def assemble_stiffness(basis, tensor, d, dofs_per_elem):
    """Assemble stiffness matrix from a per-element tensor."""
    return laplace.assemble(basis,**{f"sigma{i}{j}": DiscreteField(np.tile(tensor[..., i, j], [dofs_per_elem, 1]).T)for i, j in product(range(d), repeat=2)},d=d)

def assemble_facet_proj_op(fbasis):
    """Assemble normal projection for surface meshes."""
    normals = fbasis.normals
    normals_basis = normals[np.newaxis] * normals[:, np.newaxis]
    dim, nelems, nr_quadr_points = normals.shape
    normal_ops = []
    for quadr_i in range(nr_quadr_points):
        data = []
        rows = []
        cols = []
        for di in range(dim):
            rows.append(np.tile(np.arange(nelems), [dim, 1]).T.flatten() * dim + di)
            cols.append(np.arange(nelems*dim))
            data.append(normals_basis[di, ..., quadr_i].T.flatten())
        data, rows, cols = [np.concatenate(arr) for arr in [data, rows, cols]]
        normal_ops.append(coo_matrix((data, (rows, cols)), shape=[nelems*dim, nelems*dim]).tocsr())
        
    indices = np.arange(0,normal_ops[0].shape[0]) 
    normal_out = []
    for n_op in normal_ops:
        n_op_neg = -n_op
        n_op_neg[indices, indices] += 1 
        normal_out.append(n_op_neg)
    
    return normal_out

def assemble_quadr_grad(fbasis):
    """Assemble gradient operator for quadrature points."""
    dphi = np.stack([fbasis.basis[i][0].grad for i in range(len(fbasis.basis))]) 
    assert np.allclose(dphi.sum(0), 0.)
    nbfun, dim, nelems, nr_quadr_points = dphi.shape 
    dofs_per_elem = fbasis.element_dofs.shape[0]
    grad_ops = []
    for quadr_i in range(nr_quadr_points):
        data = []
        rows = []
        cols = []
        for di in range(dim):
            rows.append(np.tile(np.arange(nelems), [dofs_per_elem, 1]).T.flatten() * dim + di)
            cols.append(fbasis.element_dofs.T.flatten())
            data.append(dphi[:, di, :, quadr_i].T.flatten())
        
        data, rows, cols = [np.concatenate(arr) for arr in [data, rows, cols]]
        grad_ops.append(coo_matrix((data, (rows, cols)), shape=[nelems*dim, fbasis.N]).tocsr())
        
    return grad_ops


def assemble_transfer_op(torso_mesh, elec_inds, epi_inds, cond_tensors, d, heart_points):
    """Assemble the epicardium torso forward operator"""       
    if d == 3:
        elem = ElementTetP1()
    else:
        elem = ElementTriP1()

    basis = Basis(torso_mesh, elem)
    dofs_per_elem = basis.elem.doflocs.shape[0]
    
    A_orig = assemble_stiffness(basis, cond_tensors, d, dofs_per_elem)
    
    #The problem is made well-posed by enforcing the dirichlet boundary conditions on the hearts surface \Gamma_H
    pen_eps = 1e-4
    A = penalize(A_orig, D=epi_inds, epsilon=pen_eps).astype(np.float32)
    #Remove the interior points
    if heart_points is None:
        heart_points = np.array([], dtype=np.int64)

    #Assert heart_points.size == 0 or all([np.any(heart_points == epi_i) for epi_i in heart_epi_inds])
    heart_redundant_i = np.setdiff1d(heart_points, epi_inds)
    A_reduced, _,  I = condense(A, D=heart_redundant_i)
    A_reduced.eliminate_zeros()
    A_reduced = A_reduced.tocsc()

    #Avoids building the whole inverse operator
    A_chol = cholesky(A_reduced, use_long=True)


    elec_reduced_inds = np.array([np.where(e_i == I)[0][0] for e_i in elec_inds.flatten()])
    transfer_op = []
    rhs = np.zeros(shape=[A_reduced.shape[0]], dtype=np.float32)
    from tqdm import tqdm
    for i in tqdm(epi_inds):
        assert i in I
        rhs[:] = 0.
        rhs[np.where(i == I)[0]] = 1. / pen_eps
        transfer_op.append(A_chol(rhs)[elec_reduced_inds])

    transfer_op = np.stack(transfer_op, axis=1)
    return transfer_op

def quadrature_matrix_electrode(points):
    """Compute quadrature weights for a single electrode (3 points)."""
    I_1 = np.linalg.norm(points[0] - points[1])
    I_2 = np.linalg.norm(points[0] - points[2])
    vol = I_1 + I_2
    mat = np.array([(I_1 + I_2) / 2, I_1 / 2, I_2 / 2]) / vol # divide by vol for normalization
    return mat


def quadrature_matrix_all_electrodes(elec_array, torso):
    """Assemble a block-diagonal quadrature matrix for all electrodes."""
    num_electrodes = len(elec_array)
    matrix = np.zeros((num_electrodes, num_electrodes * 3))
    
    for i in range(num_electrodes):
        points_in = torso.points[elec_array[i]]
        mat = quadrature_matrix_electrode(points_in)
        matrix[i, i*3:(i+1)*3] = mat  # insert weights into the row
    
    return matrix

def build_p0_to_p1_space(fbasis, epi_inds):
    """Assemble matrix to map elementwise functions to nodewise functions."""
    mass = fbasis.dx.flatten()
    dofs_per_elem = fbasis.element_dofs.shape[0]
    
    rows = []
    cols = []

    cols = np.tile(np.arange(fbasis.nelems), [dofs_per_elem-1, 1]).T.flatten()
    rows = fbasis.element_dofs.T.flatten()
    rows_filtered = rows[np.isin(rows, epi_inds)]
    
    grad_ops = coo_matrix((mass, (rows_filtered, cols)), shape=[len(epi_inds), fbasis.nelems]).tocsr()
        
    return grad_ops

def assemble_interpol_op_space(fbasis) :
    """Assemble interpolation operator to map function values of spatial gradient to nodes."""
    phi = np.stack([fbasis.basis[i][0].value for i in range(len(fbasis.basis))]) 
    interpol_ops = []
    for phi_qp in phi.T:
        phi_qp_mask = ~np.isclose(phi_qp, 0.)
        nr_active_bases = phi_qp_mask.sum(1)[0]
        basis_i = np.where(phi_qp_mask)[1] 
        rows = np.tile(np.arange(phi_qp.shape[0]), [nr_active_bases, 1]).T.flatten() 
        cols = fbasis.element_dofs[basis_i, rows] 
        phi_qp_data = phi_qp[phi_qp_mask]
        
        phi_qp_data = np.ones_like(phi_qp_data)
        interpol_ops.append(coo_matrix((phi_qp_data, (rows, cols)), shape=(fbasis.nelems, fbasis.N)))

    interpol_op = vstack(interpol_ops).tocsr()
    boundary_nnz = np.unique(interpol_op.nonzero()[1])
    reduced_interpol_op = interpol_op[:, boundary_nnz]

    return reduced_interpol_op

def plot_circular_space_time_cylinder(vals, ax, t_all, angle, epi_order, cmap, vmin = 0, vmax = 1, ax_label = True, colorbar=False, save=None):
    """Helper function to plot a circular space-time cylinder."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6), nrows=1, ncols=1)

    fig = ax.figure
    space_time_grid = np.stack(np.meshgrid(angle[epi_order], t_all, indexing="ij"))
    cont_h = ax.contourf(*space_time_grid, vals[epi_order], levels = 24, cmap = cmap, vmin = vmin, vmax = vmax) 
    if ax_label:
        ax.set_xlabel("Angle [rad]")
        ax.set_ylabel("Time $t$ [ms]")

    if colorbar:
        cbar = fig.colorbar(cont_h, ax=ax)
    
    if save is not None:
        fig.gca().axes.get_yaxis().set_visible(False)
        fig.gca().axes.get_xaxis().set_visible(False)
        fig.savefig(save + '.png', format='png', dpi=300, transparent = True, bbox_inches = 'tight', pad_inches = 0)

def plot_space_time_comparison(vals, gt, t_all, angle, epi_order, cmap, vmin = 0, vmax = 1, ax_label = True):
    """Helper function to compare reconstruction with ground truth."""
    diff = vals - gt
    fig, axes = plt.subplots(nrows=1, ncols=4)
    plot_circular_space_time_cylinder(vals, axes[0], t_all, angle, epi_order, cmap, vmin, vmax, ax_label)
    axes[0].set_title("Reconstruction")
    plot_circular_space_time_cylinder(diff, axes[1], t_all, angle, epi_order, cmap, vmin, vmax, ax_label)
    axes[1].set_title(f"Diff")
    plot_circular_space_time_cylinder(gt, axes[2], t_all, angle, epi_order, vmin, cmap, vmax, ax_label)
    axes[2].set_title("GT")
    axes[3].hist(np.abs(diff).flatten(), bins=100, density=True)
    axes[3].set_title("Error distribution")

    fig.set_size_inches((18, 7))
    return fig, axes

