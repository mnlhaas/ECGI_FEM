import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.spatial import cKDTree
import pyvista as pv
import meshio
import pandas as pd
from pathlib import Path

from itertools import product
import multiprocessing
import matplotlib.pyplot as plt

from skfem import Mesh, MeshTet, ElementTetP1, ElementTriP1, Basis, DiscreteField, Element, penalize, condense
from skfem.io import from_meshio
from skfem import BilinearForm
from skfem.helpers import dot, grad
from sksparse.cholmod import cho_factor


def compute_normalization_stats(dim):
    """Compute global, train-set-only min/max normalization stats.
    """
    data_dir = Path(f"data/{dim}")
    train_files = pd.read_csv(data_dir / "data_csv" / "train.csv", header=None).to_numpy().squeeze(axis=1)

    u_min, u_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    for fname in train_files:
        d = np.load(data_dir / "data_functions" / fname)
        u_min = min(u_min, d["u"].min(), d["u_fine"].min())
        u_max = max(u_max, d["u"].max(), d["u_fine"].max())
        y_min = min(y_min, d["y"].min())
        y_max = max(y_max, d["y"].max())

    out_path = data_dir / "data_fixed" / "normalization.npz"
    np.savez(out_path, u_min=u_min, u_max=u_max, y_min=y_min, y_max=y_max)
    print(f"normalization stats ({dim}): u=[{u_min:.4f},{u_max:.4f}] y=[{y_min:.4f},{y_max:.4f}] -> {out_path}", flush=True)
    
def farthest_point_sampling(points, n_samples, seed=42):
    """Greedy farthest-point sampling among a discrete candidate point set."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    selected = np.empty(n_samples, dtype=int)
    selected[0] = rng.integers(n)
    dist = np.linalg.norm(points - points[selected[0]], axis=1)
    for i in range(1, n_samples):
        selected[i] = np.argmax(dist)
        dist = np.minimum(dist, np.linalg.norm(points - points[selected[i]], axis=1))
    return selected

def build_interp_matrix(elem_idx, bary_weights, elem_node_idx, n_query, n_source):
    """Assemble a sparse (n_query, n_source) interpolation matrix from, for each
    query point, the index of its containing/closest element and its
    barycentric weights over that element's nodes.

    elem_idx: (n_query,) index into `elem_node_idx`'s first axis (which element
        each query point belongs to).
    bary_weights: (n_query, n_nodes_per_elem) barycentric weights.
    elem_node_idx: (n_elem, n_nodes_per_elem) node indices per element (into the
        source point array of size n_source).
    """
    nodes = elem_node_idx[elem_idx]  # (n_query, n_nodes_per_elem)
    n_query_, n_npe = nodes.shape
    rows = np.repeat(np.arange(n_query_), n_npe)
    cols = nodes.flatten()
    data = bary_weights.flatten()
    return coo_matrix((data, (rows, cols)), shape=(n_query, n_source)).tocsr()


def locate_in_tets(query_points, mesh_points, tets, k_start=16, k_max=1024, tol=1e-3):
    """Locate each query point within a tetrahedral mesh and return barycentric
    interpolation data.
    """
    query_points = np.asarray(query_points)
    n_query = query_points.shape[0]
    centroids = mesh_points[tets].mean(axis=1)
    tree = cKDTree(centroids)

    best_tet = np.full(n_query, -1, dtype=np.int64)
    best_bary = np.zeros((n_query, 4))
    best_violation = np.full(n_query, np.inf)

    unresolved = np.arange(n_query)
    k = k_start
    while unresolved.size > 0:
        k = min(k, tets.shape[0])
        _, cand = tree.query(query_points[unresolved], k=k)
        if k == 1:
            cand = cand[:, None]
        for j in range(cand.shape[1]):
            tet_idx = cand[:, j]
            verts = mesh_points[tets[tet_idx]]  # (n_unresolved, 4, 3)
            v0 = verts[:, 0]
            T = np.stack([verts[:, 1] - v0, verts[:, 2] - v0, verts[:, 3] - v0], axis=-1)  # (n, 3, 3)
            rhs = query_points[unresolved] - v0
            b123 = np.linalg.solve(T, rhs[..., None])[..., 0]  # (n, 3)
            b0 = 1.0 - b123.sum(axis=1)
            bary = np.concatenate([b0[:, None], b123], axis=1)
            violation = np.clip(-bary, 0, None).max(axis=1)
            improve = violation < best_violation[unresolved]
            idx_glob = unresolved[improve]
            best_tet[idx_glob] = tet_idx[improve]
            best_bary[idx_glob] = bary[improve]
            best_violation[idx_glob] = violation[improve]

        unresolved = unresolved[best_violation[unresolved] > tol]
        if k >= k_max or k >= tets.shape[0]:
            break
        k *= 4

    missing_mask = best_violation > tol
    bary = np.clip(best_bary, 0, None)
    bary /= bary.sum(axis=1, keepdims=True)
    return best_tet, bary, missing_mask


def _closest_point_on_triangles(p, a, b, c):
    """Vectorized closest-point-on-triangle (Ericson, Real-Time Collision
    Detection). p, a, b, c: (n, 3). Returns closest (n,3), bary (n,3)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)

    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)

    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    n = p.shape[0]
    bary = np.zeros((n, 3))
    closest = np.zeros((n, 3))
    unresolved = np.ones(n, dtype=bool)

    case1 = unresolved & (d1 <= 0) & (d2 <= 0)
    bary[case1] = np.array([1.0, 0.0, 0.0])
    closest[case1] = a[case1]
    unresolved &= ~case1

    case2 = unresolved & (d3 >= 0) & (d4 <= d3)
    bary[case2] = np.array([0.0, 1.0, 0.0])
    closest[case2] = b[case2]
    unresolved &= ~case2

    case3 = unresolved & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v = np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        v[case3] = d1[case3] / (d1[case3] - d3[case3])
    bary[case3] = np.stack([1 - v[case3], v[case3], np.zeros(case3.sum())], axis=1)
    closest[case3] = a[case3] + v[case3, None] * ab[case3]
    unresolved &= ~case3

    case4 = unresolved & (d6 >= 0) & (d5 <= d6)
    bary[case4] = np.array([0.0, 0.0, 1.0])
    closest[case4] = c[case4]
    unresolved &= ~case4

    case5 = unresolved & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w = np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        w[case5] = d2[case5] / (d2[case5] - d6[case5])
    bary[case5] = np.stack([1 - w[case5], np.zeros(case5.sum()), w[case5]], axis=1)
    closest[case5] = a[case5] + w[case5, None] * ac[case5]
    unresolved &= ~case5

    case6 = unresolved & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    w6 = np.zeros(n)
    denom6 = (d4 - d3) + (d5 - d6)
    with np.errstate(invalid="ignore", divide="ignore"):
        w6[case6] = (d4[case6] - d3[case6]) / denom6[case6]
    bary[case6] = np.stack([np.zeros(case6.sum()), 1 - w6[case6], w6[case6]], axis=1)
    closest[case6] = b[case6] + w6[case6, None] * (c[case6] - b[case6])
    unresolved &= ~case6

    # interior
    denom = va + vb + vc
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(unresolved, vb / np.where(denom == 0, 1, denom), 0.0)
        w = np.where(unresolved, vc / np.where(denom == 0, 1, denom), 0.0)
    bary[unresolved] = np.stack([1 - v[unresolved] - w[unresolved], v[unresolved], w[unresolved]], axis=1)
    closest[unresolved] = a[unresolved] + v[unresolved, None] * ab[unresolved] + w[unresolved, None] * ac[unresolved]

    return closest, bary


def locate_on_triangles(query_points, surf_points, tris, k=12):
    """For each query point, find the closest point on a triangulated surface
    (embedded in 3D) among the `k` nearest candidate triangles (by centroid),
    and return barycentric interpolation data for that closest point.
    """
    query_points = np.asarray(query_points)
    n_query = query_points.shape[0]
    centroids = surf_points[tris].mean(axis=1)
    tree = cKDTree(centroids)
    k = min(k, tris.shape[0])
    _, cand = tree.query(query_points, k=k)
    if k == 1:
        cand = cand[:, None]

    best_tri = np.full(n_query, -1, dtype=np.int64)
    best_bary = np.zeros((n_query, 3))
    best_dist = np.full(n_query, np.inf)

    for j in range(cand.shape[1]):
        tri_idx = cand[:, j]
        verts = surf_points[tris[tri_idx]]  # (n_query, 3, 3)
        closest, bary = _closest_point_on_triangles(
            query_points, verts[:, 0], verts[:, 1], verts[:, 2]
        )
        dist = np.linalg.norm(query_points - closest, axis=1)
        improve = dist < best_dist
        best_tri[improve] = tri_idx[improve]
        best_bary[improve] = bary[improve]
        best_dist[improve] = dist[improve]

    return best_tri, best_bary, best_dist


def remesh_torso_coarser(outer_surface, inner_surface, mode="region",region_seed_point=None, torso_seed_point=None, max_volume=None):
    """Build a deliberately coarser, independent volumetric tet mesh of the
    torso"""
    import tetgen

    merged = pv.merge([outer_surface, inner_surface])
    tgen = tetgen.TetGen(merged)
    quality = max_volume is not None

    if mode == "region":
        tgen.add_region(1, list(region_seed_point), max_vol=(max_volume or 0.0))
        tgen.add_region(2, list(torso_seed_point), max_vol=(max_volume or 0.0))
        nodes, elems, attrib, _ = tgen.tetrahedralize(
            plc=True, nobisect=True, quality=quality, regionattrib=True, varvolume=quality, verbose=0
        )
        attrib = attrib.flatten()
        centroids = nodes[elems].mean(axis=1)
        nearest_elem = np.argmin(np.linalg.norm(centroids - np.asarray(region_seed_point), axis=1))
        interior_marker = attrib[nearest_elem]
        interior_elem_idx = np.where(attrib == interior_marker)[0]
        mesh = MeshTet(nodes.T, elems.T, _subdomains={"interior": interior_elem_idx})
    elif mode == "hole":
        tgen.add_hole(list(region_seed_point))
        if quality:
            result = tgen.tetrahedralize(switches=f"pYQa{max_volume}")
        else:
            result = tgen.tetrahedralize(plc=True, nobisect=True, quality=False, verbose=0)
        nodes, elems = result[0], result[1]
        mesh = MeshTet(nodes.T, elems.T)
    else:
        raise ValueError(f"mode must be 'region' or 'hole', got {mode!r}")

    return mesh


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
    A_chol = cho_factor(A_reduced)


    elec_reduced_inds = np.array([np.where(e_i == I)[0][0] for e_i in elec_inds.flatten()])
    transfer_op = []
    rhs = np.zeros(shape=[A_reduced.shape[0]], dtype=np.float32)
    from tqdm import tqdm
    for i in tqdm(epi_inds):
        assert i in I
        rhs[:] = 0.
        rhs[np.where(i == I)[0]] = 1. / pen_eps
        transfer_op.append(A_chol.solve(rhs)[elec_reduced_inds])

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

def assemble_electrode_patches_3d(points, boundary_facets, facets, elec_inds):
    """Assemble electrode patches around the electrode nodes and compute averaged integral of those
    """
    from scipy.sparse import lil_matrix

    tri_nodes = facets[:, boundary_facets].T  # (n_boundary_tris, 3)
    tri_pts = points[tri_nodes]  # (n_boundary_tris, 3, 3)
    e1 = tri_pts[:, 1] - tri_pts[:, 0]
    e2 = tri_pts[:, 2] - tri_pts[:, 0]
    tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    node_to_tris = {}
    for t_idx, tri in enumerate(tri_nodes):
        for n in tri:
            node_to_tris.setdefault(n, []).append(t_idx)

    all_patch_nodes = set()
    elec_nodes, elec_weights = [], []
    for center in elec_inds:
        node_weight = {}
        for t_idx in node_to_tris.get(center, []):
            tri, area = tri_nodes[t_idx], tri_area[t_idx]
            for n in tri:
                node_weight[n] = node_weight.get(n, 0.0) + area / 3.0
        total = sum(node_weight.values())

        nodes = list(node_weight.keys())
        weights = [node_weight[n] / total for n in nodes]
        elec_nodes.append(nodes)
        elec_weights.append(weights)
        all_patch_nodes.update(nodes)

    patch_node_inds = np.array(sorted(all_patch_nodes))
    node_pos = {n: i for i, n in enumerate(patch_node_inds)}

    quad_matrix = lil_matrix((len(elec_inds), len(patch_node_inds)))
    for i, (nodes, weights) in enumerate(zip(elec_nodes, elec_weights)):
        for n, w in zip(nodes, weights):
            quad_matrix[i, node_pos[n]] = w

    return patch_node_inds, quad_matrix.tocsr()

def build_p0_to_p1_space(fbasis, epi_inds):
    """Assemble matrix to map elementwise functions to nodewise functions."""
    mass = fbasis.dx.flatten()
    facet_nodes = fbasis.mesh.facets[:, fbasis.find]  # (dofs_per_facet, nelems)

    cols = np.repeat(np.arange(fbasis.nelems), facet_nodes.shape[0])
    rows_global = facet_nodes.T.flatten()

    epi_pos = -np.ones(fbasis.mesh.p.shape[1], dtype=np.int64)
    epi_pos[epi_inds] = np.arange(len(epi_inds))
    rows = epi_pos[rows_global]
    assert (rows >= 0).all(), "facet basis contains nodes outside epi_inds"

    grad_ops = coo_matrix((mass, (rows, cols)), shape=[len(epi_inds), fbasis.nelems]).tocsr()

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

