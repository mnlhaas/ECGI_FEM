import json
import os
import contextlib
import numpy as np
import h5py
import pyvista as pv
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import save_npz, load_npz, csr_matrix
from sksparse.cholmod import cho_factor

from skfem import ElementTetP1, FacetBasis, Basis, MeshTet

from utils_data import (
    mass,
    assemble_stiffness,
    assemble_facet_proj_op,
    assemble_quadr_grad,
    assemble_transfer_op,
    assemble_electrode_patches_3d,
    build_p0_to_p1_space,
    assemble_interpol_op_space,
    compute_normalization_stats,
    farthest_point_sampling,
    remesh_torso_coarser,
    locate_in_tets,
    build_interp_matrix,
)


class GenData3D:
    """
    Data generator for 3D cardiac electrophysiology simulations on a biventricular
    heart mesh with rule-based (LDRB) fiber directions.
    """

    PACING_SITES = ["LV", "RV", "APEX"]

    def __init__(self, config):
        self.config = config
        self.data_nb = config["data_nb"]
        
        # Stimulation parameters
        self.Imax = config["stimulation"]["Imax"]
        self.Idur = config["stimulation"]["Idur"]
        
        # Membrane parameters
        self.Cm = config["membrane"]["Cm"]
        self.beta = config["membrane"]["beta"]
        
        # Time discretization
        self.Tend = config["time"]["Tend"]
        self.dt_range = config["time"]["dt_range"]
        self.sample_range = config["time"]["sample_range"]
        
        # Ionic model parameters (Nagumo, no repolarization)
        self.Vrest = config["ionic_model"]["Vrest"]
        self.Vdep = config["ionic_model"]["Vdep"]
        self.Vthre = config["ionic_model"]["Vthre"]
        self.gmax = config["ionic_model"]["gmax"]
        self.n_electrodes = config["electrodes"]["n_electrodes"]
        self.sigma_torso = config["torso"]["sigma_torso"]
        
        self.d = 3
        
        mesh_dir = Path("data/meshes/3D")

        self.MM_TO_CM = 0.1

        # Full torso and heart domain
        with h5py.File(mesh_dir / "mesh_000.h5", "r") as f:
            self.points = f["data0"][:] * self.MM_TO_CM
            self.tets = f["data1"][:]
            self.region = f["data2"][:]
        heart_elem_idx = np.where(self.region == 1)[0]

        self.big_mesh = MeshTet(
            self.points.T, self.tets.T,
            _subdomains={"heart": heart_elem_idx},
        )

        # Heart-only mesh with LDRB fibers
        heart_pv = pv.UnstructuredGrid(str(mesh_dir / "mesh_000.colored.vtu"))
        self.heart_pts = heart_pv.points * self.MM_TO_CM
        self.heart_tets = heart_pv.cells_dict[pv.CellType.TETRA]
        self.fibers_node = heart_pv.point_data["Fibers"]
        ldrb_cell = heart_pv.cell_data["LDRB"]
        apex_base = heart_pv.point_data["apex_base"]
        lv = heart_pv.point_data["lv"]
        ldrb = heart_pv.point_data["LDRB"]

        self.pacing_site_inds = np.array([self.get_pacing_site(self.heart_pts, apex_base, lv, ldrb, site) for site in self.PACING_SITES])

        self.heart = MeshTet(self.heart_pts.T, self.heart_tets.T)

        # Local to global index correspondence
        big_heart_node_idx = np.unique(self.tets[heart_elem_idx])
        tree = cKDTree(self.points[big_heart_node_idx])
        dist, order = tree.query(self.heart_pts, k=1)
        assert dist.max() < 1e-4, f"heart<->torso mesh coordinate mismatch: {dist.max()}"
        self.local_to_global = big_heart_node_idx[order]

        # Config's scar/stim radii are fractions of heart size
        heart_diag = np.linalg.norm(self.heart_pts.max(0) - self.heart_pts.min(0))
        self.stim_radius = config["stim_radius_frac"] * heart_diag
        self.scar_rad_range = [f * heart_diag for f in config["scar"]["rad_frac"]]
        self.scar_margin = config["stim_radius_frac"] * heart_diag

        self.c = self.heart_pts[self.heart_tets].mean(axis=1)

        # Epicardial surface (coarse)
        boundary_facets = self.heart.boundary_facets()
        facet_nodes = self.heart.facets[:, boundary_facets]
        owner_tets = self.heart.f2t[0, boundary_facets]
        epi_facet_mask = np.isin(ldrb_cell[owner_tets], [4, 7])
        epi_facets = boundary_facets[epi_facet_mask]

        # Nodes of epicardial facets
        self.is_epi_node = np.zeros(self.heart_pts.shape[0], dtype=bool)
        self.is_epi_node[np.unique(facet_nodes[:, epi_facet_mask])] = True

        self.heart_surf_basis = FacetBasis(self.heart, ElementTetP1(), facets=epi_facets)
        self.epi_inds = self.heart_surf_basis.get_dofs(facets=epi_facets).all()

        self.epi_points = self.heart_pts[self.epi_inds]
        epi_remap = -np.ones(self.heart_pts.shape[0], dtype=np.int64)
        epi_remap[self.epi_inds] = np.arange(self.epi_inds.size)
        self.epi_tris = epi_remap[self.heart.facets[:, epi_facets]].T  # (n_facets, 3)


        self.full_surf_basis = FacetBasis(self.heart, ElementTetP1(), facets=boundary_facets)
        self.surf_inds = self.full_surf_basis.get_dofs(facets=boundary_facets).all()
        self.surf_points = self.heart_pts[self.surf_inds]
        surf_remap = -np.ones(self.heart_pts.shape[0], dtype=np.int64)
        surf_remap[self.surf_inds] = np.arange(self.surf_inds.size)
        self.surf_tris = surf_remap[self.heart.facets[:, boundary_facets]].T  # (n_boundary_facets, 3)
        self.epi_tris_in_surf = surf_remap[self.heart.facets[:, epi_facets]].T

        self.heart_basis = Basis(self.heart, ElementTetP1())
        self.dofs_per_elem = self.heart_basis.elem.doflocs.shape[0]
        fibers_elem = self.fibers_node[self.heart_tets].mean(axis=1)
        self.l = fibers_elem / np.linalg.norm(fibers_elem, axis=1, keepdims=True)
        self.I = np.repeat(np.eye(3)[None, :, :], self.c.shape[0], axis=0)

        # Finer mesh used only for the monodomain solve
        refine_levels = config.get("sim_mesh_refine_levels", 0)
        if refine_levels > 0:
            self.heart_sim = self.heart.refined(refine_levels)
            self.heart_sim_pts = self.heart_sim.p.T
            self.heart_sim_tets = self.heart_sim.t.T
            assert np.allclose(self.heart_pts, self.heart_sim_pts[:self.heart_pts.shape[0]]), \
                "refined mesh does not preserve original nodes as an exact prefix"
            self.orig_in_sim_idx = np.arange(self.heart_pts.shape[0])

            self.c_sim = self.heart_sim_pts[self.heart_sim_tets].mean(axis=1)
            tree_c = cKDTree(self.c)
            _, nn_sim = tree_c.query(self.c_sim)
            self.l_sim = self.l[nn_sim]
            self.I_sim = np.repeat(np.eye(3)[None, :, :], self.c_sim.shape[0], axis=0)

            self.heart_sim_basis = Basis(self.heart_sim, ElementTetP1())
            self.dofs_per_elem_sim = self.heart_sim_basis.elem.doflocs.shape[0]
        else:
            self.heart_sim = self.heart
            self.heart_sim_pts = self.heart_pts
            self.heart_sim_tets = self.heart_tets
            self.orig_in_sim_idx = np.arange(self.heart_pts.shape[0])
            self.c_sim = self.c
            self.l_sim = self.l
            self.I_sim = self.I
            self.heart_sim_basis = self.heart_basis
            self.dofs_per_elem_sim = self.dofs_per_elem

        # Electrodes placement on torso
        big_boundary_facets = self.big_mesh.boundary_facets()
        outer_node_idx = np.unique(self.big_mesh.facets[:, big_boundary_facets])
        heart_z_lo, heart_z_hi = self.heart_pts[:, 2].min(), self.heart_pts[:, 2].max()
        z_margin = 1.5 * (heart_z_hi - heart_z_lo)
        z_lo, z_hi = heart_z_lo - z_margin, heart_z_hi + z_margin
        heart_xy = self.heart_pts[:, :2].mean(0)
        radius_max = 20.0
        xy_dist = np.linalg.norm(self.points[outer_node_idx, :2] - heart_xy, axis=1)
        torso_mask = ((self.points[outer_node_idx, 2] >= z_lo) & (self.points[outer_node_idx, 2] <= z_hi)& (xy_dist <= radius_max))
        outer_node_idx = outer_node_idx[torso_mask]
        fps_order = farthest_point_sampling(self.points[outer_node_idx], self.n_electrodes)
        self.elec_inds = outer_node_idx[fps_order]

        # Coarse torso+heart mesh used for the inverse problem's discretization
        self.build_sparse_mesh()

        if self.config["plot"] or self.config.get("n_plot_samples", 0) > 0:
            from matplotlib.colors import ListedColormap
            from scipy.interpolate import interp1d
            import pandas as pd
    
            lut_vals = 1024
            rgb = pd.read_csv("data/colormap/coolwarm_extended.csv")
            rgba = np.concatenate([rgb.to_numpy() / 255, np.ones([rgb.shape[0], 1])], axis=1)
            rgba_interp = interp1d(np.linspace(0, 1, num=rgba.shape[0]), rgba.T)(np.linspace(0, 1, num=lut_vals)).T
            self.cmap_new = ListedColormap(rgba_interp)


    def build_sparse_mesh(self):
        """Build the coarse torso+heart mesh used for the inverse problem's discretization"""
        mesh_dir = Path("data/meshes/3D")
        sparse_poly = pv.read(str(mesh_dir / "mesh_000_closed.stl"))
        sparse_poly.points = sparse_poly.points * self.MM_TO_CM
        self.sparse_points = sparse_poly.points
        self.sparse_tris = sparse_poly.faces.reshape(-1, 4)[:, 1:]

        big_boundary_facets = self.big_mesh.boundary_facets()
        outer_tris = self.big_mesh.facets[:, big_boundary_facets].T
        outer_surf = pv.PolyData(
            self.points, np.hstack([np.full((len(outer_tris), 1), 3), outer_tris]).astype(np.int64)
        ).clean()

        heart_interior_seed = self.heart_pts[self.heart_tets[0]].mean(axis=0)
        self.coarse_sparse_epi = remesh_torso_coarser(
            outer_surf, sparse_poly, mode="hole", region_seed_point=heart_interior_seed, max_volume=1.2,
        )
        self.coarse_sparse_pts = self.coarse_sparse_epi.p.T

        tree = cKDTree(self.coarse_sparse_pts)
        dist, epi_inds_sparse_in_coarse = tree.query(self.sparse_points, k=1)
        assert dist.max() < 1e-6, f"sparse-epicardium node mismatch in coarse mesh: {dist.max()}"
        self.epi_inds_sparse_in_coarse = epi_inds_sparse_in_coarse

        is_heart_hole_point = np.zeros(self.coarse_sparse_pts.shape[0], dtype=bool)
        is_heart_hole_point[epi_inds_sparse_in_coarse] = True
        coarse_sparse_boundary_facets = self.coarse_sparse_epi.boundary_facets()
        facet_nodes = self.coarse_sparse_epi.facets[:, coarse_sparse_boundary_facets]
        heart_hole_mask = is_heart_hole_point[facet_nodes].all(axis=0)
        self.sparse_heart_hole_facets = coarse_sparse_boundary_facets[heart_hole_mask]
        coarse_sparse_outer_facets = coarse_sparse_boundary_facets[~heart_hole_mask]

        heart_hole_facet_nodes = self.coarse_sparse_epi.facets[:, self.sparse_heart_hole_facets]
        heart_hole_centroids = self.coarse_sparse_pts[heart_hole_facet_nodes].mean(axis=0)
        tri_centroids = self.sparse_points[self.sparse_tris].mean(axis=1)
        tree = cKDTree(heart_hole_centroids)
        dist, reorder = tree.query(tri_centroids, k=1)
        self.sparse_heart_hole_facets = self.sparse_heart_hole_facets[reorder]

        self.sparse_surf_basis = FacetBasis(self.coarse_sparse_epi, ElementTetP1(), facets=self.sparse_heart_hole_facets)
        self.sparse_inds = self.sparse_surf_basis.get_dofs(facets=self.sparse_heart_hole_facets).all()

        # Electrode patches on the coarse torso's own outer boundary
        coarse_sparse_outer_node_idx = np.unique(self.coarse_sparse_epi.facets[:, coarse_sparse_outer_facets])
        elec_tree = cKDTree(self.coarse_sparse_pts[coarse_sparse_outer_node_idx])
        _, nearest_local = elec_tree.query(self.points[self.elec_inds], k=1)
        self.elec_inds_coarse_sparse = coarse_sparse_outer_node_idx[nearest_local]

    def fion(self, vm):
        """Nagumo ionic current model"""
        return (vm - self.Vrest) * (vm - self.Vthre) * (vm - self.Vdep) * self.gmax

    def get_pacing_site(self, pts, apex_base, lv, ldrb, site):
        """Return the heart-mesh node index for one of the three anatomical pacing sites (LV, RV, APEX)
        """
        if site == "APEX":
            return int(np.argmin(apex_base))

        elif site == "LV":
            rv_ind = self.get_pacing_site(pts, apex_base, lv, ldrb, "RV")
            mask_lv = ldrb == 4
            if mask_lv.sum() == 0:
                mask_lv = lv == 1
            mask_rv = (lv == 0) & (ldrb == 0)
            if mask_rv.sum() == 0:
                mask_rv = lv == 0
            height_tol = 0.15 * (pts[:, 0].max() - pts[:, 0].min())
            height_mask = np.abs(pts[:, 0] - pts[rv_ind, 0]) < height_tol
            cands = np.where(mask_lv & height_mask)[0]
            if cands.size == 0:
                cands = np.where(mask_lv)[0]
            rv_tree = cKDTree(pts[mask_rv])
            dists, _ = rv_tree.query(pts[cands], k=1)
            return int(cands[np.argmax(dists)])

        elif site == "RV":
            mask_rv = (lv == 0) & (ldrb == 0)
            if mask_rv.sum() == 0:
                mask_rv = lv == 0
            mask_lv = lv == 1
            cands = np.where(mask_rv)[0]
            lv_tree = cKDTree(pts[mask_lv])
            dists, _ = lv_tree.query(pts[cands], k=1)
            return int(cands[np.argmax(dists)])

        elif site == "RVOT":
            mask = (lv == 0) & (ldrb == 0) & (apex_base > 0.65)
            if mask.sum() == 0:
                mask = (lv == 0) & (ldrb == 0) & (apex_base > 0.5)
            cands = np.where(mask)[0]
            return int(cands[np.argmin(np.abs(apex_base[cands] - 0.80))])

        else:
            raise ValueError(f"Unknown pacing site: {site!r}. Choose from {self.PACING_SITES}")

    def plot_potential_snapshots(self, points, tris, potential, stim_pos, out_path, vmin=None, vmax=None, colorbar_label="Extracellular potential [mV]"):
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        tt = np.linspace(0, potential.shape[1] - 1, 9).astype(int)
        if vmin is None:
            vmin = potential.min()
        if vmax is None:
            vmax = potential.max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        fig = plt.figure(figsize=(16, 3))
        for k, i in enumerate(tt[2:-2]):
            ax = fig.add_subplot(1, 5, k + 1, projection="3d")
            face_vals = potential[tris, i].mean(axis=1)
            poly = Poly3DCollection(points[tris], facecolor=self.cmap_new(norm(face_vals)))
            ax.add_collection3d(poly)
            if stim_pos is not None:
                ax.scatter(*stim_pos, color="black", s=25, depthshade=False)
            ax.set_xlim(points[:, 0].min(), points[:, 0].max())
            ax.set_ylim(points[:, 1].min(), points[:, 1].max())
            ax.set_zlim(points[:, 2].min(), points[:, 2].max())
            ax.axis("off")

        sm = cm.ScalarMappable(cmap=self.cmap_new, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=fig.axes, orientation="vertical", pad=0.02, fraction=0.05, label=colorbar_label)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    def gen_sample(self, seed):
        """Generate one simulation sample of cardiac potentials"""
        np.random.seed(seed)

        # Conductivities (intra- and extracellular)
        lamb_LT = np.random.uniform(self.config["bidomain_cond"]["lamb_LT"][0], self.config["bidomain_cond"]["lamb_LT"][1])
        eps = np.random.uniform(self.config["bidomain_cond"]["eps"][0], self.config["bidomain_cond"]["eps"][1])

        alpha = self.config["bidomain_cond"]["alpha"]
        sigma_il = self.config["bidomain_cond"]["sigma_il"]

        sigma_it = sigma_il * (1 / lamb_LT) ** 2 * ((1 + alpha * (1 - eps)) / (1 + alpha))
        sigma_el = sigma_il * 1 / alpha
        sigma_et = sigma_it * 1 / (alpha * (1 - eps))

        # Intracellular / extracellular conductivity tensors on the coarse mesh
        G_i = sigma_it * self.I + (sigma_il - sigma_it) * self.l[:, :, None] @ self.l[:, None, :]
        G_e = sigma_et * self.I + (sigma_el - sigma_et) * self.l[:, :, None] @ self.l[:, None, :]
        G_i_sim = sigma_it * self.I_sim + (sigma_il - sigma_it) * self.l_sim[:, :, None] @ self.l_sim[:, None, :]
        G_e_sim = sigma_et * self.I_sim + (sigma_el - sigma_et) * self.l_sim[:, :, None] @ self.l_sim[:, None, :]

        # Stimulate from one of the fixed anatomical pacing sites (LV, RV, APEX)
        stim_center = self.pacing_site_inds[np.random.randint(len(self.pacing_site_inds))]
        Istim_sim = (np.linalg.norm(self.heart_sim_pts - self.heart_pts[stim_center], axis=1) < self.stim_radius).astype(float)

        # Generate scar tissue away from stimulation site (applied identically to both meshes)
        self.scar_info = []
        if np.random.rand() < self.config["scar"]["prob"]:
            G_i, G_e, G_i_sim, G_e_sim, scar_center_in, scar_radius_in = self.gen_scar(stim_center, G_i, G_e, G_i_sim, G_e_sim)
            self.scar_info.append((scar_center_in, scar_radius_in))
            if np.random.rand() < self.config["scar"]["prob_second"]:
                G_i, G_e, G_i_sim, G_e_sim, scar_center_in2, scar_radius_in2 = self.gen_scar(
                    stim_center, G_i, G_e, G_i_sim, G_e_sim, scar_center_in, scar_radius_in,
                )
                self.scar_info.append((scar_center_in2, scar_radius_in2))

        # Monodomain effective conductivity on the refined simulation mesh (G_m = G_i * (G_i+G_e)^-1 * G_e)
        G_sum_sim = G_i_sim + G_e_sim
        G_m_sim = np.einsum("nij,njk,nkl->nil", G_i_sim, np.linalg.inv(G_sum_sim), G_e_sim)

        # Assemble FEM mass/stiffness on the refined simulation mesh and solve the monodomain PDE there
        M_sim = mass.assemble(self.heart_sim_basis)
        K_sim = assemble_stiffness(self.heart_sim_basis, G_m_sim, self.d, self.dofs_per_elem_sim)

        time_sample = np.random.randint(self.sample_range[0], self.sample_range[1])
        dt = np.random.uniform(self.dt_range[0], self.dt_range[1])
        ndt = int(np.rint(self.Tend / dt)) + 1

        A_sim = (M_sim * self.Cm + K_sim * dt / self.beta).tocsc()
        solver_sim = cho_factor(A_sim)

        # Initial condition: resting potential
        u = np.full(self.heart_sim_pts.shape[0], self.Vrest)
        u_hist_sim = [u.copy()]
        for i in range(1, ndt):
            t = i * dt

            # Ionic current and stimulation
            Iion = self.fion(u)
            Is = Istim_sim * (self.Imax if t < self.Idur else 0.0)

            # Backward Euler step
            b = M_sim @ (u * self.Cm - (Iion - Is) * dt)
            u = solver_sim.solve(b)

            if i % time_sample == 0:
                u_hist_sim.append(u.copy())

        u_hist_sim = np.array(u_hist_sim).T

        # Restrict the refined-mesh solution down to the original discretization
        u_hist = u_hist_sim[self.orig_in_sim_idx]

        # Extracellular potential on the coarse heart mesh
        M = mass.assemble(self.heart_basis)
        K_i = assemble_stiffness(self.heart_basis, G_i, self.d, self.dofs_per_elem)
        K_e = assemble_stiffness(self.heart_basis, G_e, self.d, self.dofs_per_elem)

        epsilon = 1e-9
        A_e = (K_i + K_e + epsilon * M).tocsc()
        solver_e = cho_factor(A_e)

        rhs_e = np.asarray(-K_i @ u_hist)
        extra = solver_e.solve(rhs_e)
        epi_potential = extra[self.surf_inds]

        self.epi_potential_fine = epi_potential

        if self.config["plot"]:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from meshio.xdmf import TimeSeriesWriter

            full_facets = self.heart.boundary_facets()
            full_tris = self.heart.facets[:, full_facets].T  # (n_facets, 3) node indices, full closed surface
            pts = self.heart_pts

            self.plot_potential_snapshots(
                pts, full_tris, extra, pts[stim_center],
                f"data/3D/plots/extracellular_potential_{seed}.png",
                vmin=self.epi_potential_fine.min(), vmax=self.epi_potential_fine.max(),
            )

            epi_facets = self.heart_surf_basis.find
            epi_tris = self.heart.facets[:, epi_facets].T  # (n_facets, 3) node indices
            epi_owner_tets = self.heart.f2t[0, epi_facets]
            scar_facet_mask = np.zeros(epi_facets.size, dtype=bool)
            for scar_center, scar_radius in self.scar_info:
                scar_facet_mask |= np.linalg.norm(self.c[epi_owner_tets] - self.c[scar_center], axis=1) < scar_radius

            fig2 = plt.figure(figsize=(5, 5))
            ax2 = fig2.add_subplot(111, projection="3d")
            facecolor = np.where(scar_facet_mask, "firebrick", "lightgray")
            poly2 = Poly3DCollection(pts[epi_tris], facecolor=facecolor, edgecolor="none")
            ax2.add_collection3d(poly2)
            ax2.scatter(*self.heart_pts[stim_center], color="black", s=40, depthshade=False, label="pacing site")
            ax2.set_xlim(pts[:, 0].min(), pts[:, 0].max())
            ax2.set_ylim(pts[:, 1].min(), pts[:, 1].max())
            ax2.set_zlim(pts[:, 2].min(), pts[:, 2].max())
            ax2.axis("off")
            n_scars = len(self.scar_info)
            ax2.set_title(f"{n_scars} scar(s)" if n_scars else "no scar this sample")
            ax2.legend(loc="upper left")
            plt.savefig(f"data/3D/plots/scar_location_{seed}.png", dpi=150)
            plt.close(fig2)

            xdmf_dir = Path("data/3D/plots/xdmf")
            xdmf_dir.mkdir(parents=True, exist_ok=True)
            times = np.arange(extra.shape[1]) * time_sample * dt

            with contextlib.chdir(xdmf_dir):
                with TimeSeriesWriter(f"volume_potential_{seed}.xdmf") as writer:
                    writer.write_points_cells(self.heart_pts, [("tetra", self.heart_tets)])
                    for i in range(extra.shape[1]):
                        writer.write_data(float(times[i]), point_data={"u": extra[:, i].astype(np.float32)})

                with TimeSeriesWriter(f"epi_potential_{seed}.xdmf") as writer:
                    writer.write_points_cells(self.surf_points, [("triangle", self.surf_tris)])
                    for i in range(epi_potential.shape[1]):
                        writer.write_data(float(times[i]), point_data={"u_epi": epi_potential[:, i].astype(np.float32)})

        return epi_potential, time_sample * dt

    def gen_scar(self, stim_center, G_i, G_e, G_i_sim, G_e_sim, scar_center_in=None, scar_radius_in=None):
        """Generate scar tissue with reduced conductivity."""
        scar_radius = np.random.uniform(self.scar_rad_range[0], self.scar_rad_range[1])
        while True:
            scar_center = np.random.randint(0, self.c.shape[0])
            dist = np.linalg.norm(self.c[scar_center] - self.heart_pts[stim_center])
            if scar_center_in is not None:
                dist_scar = np.linalg.norm(self.c[scar_center] - self.c[scar_center_in])
                check_scar_dist = dist_scar > scar_radius_in + self.scar_margin
            else:
                check_scar_dist = True
            if dist > scar_radius + self.scar_margin and check_scar_dist:
                break

        scar_point = self.c[scar_center]
        scar_factor = np.random.uniform(self.config["scar"]["cond_factor"][0], self.config["scar"]["cond_factor"][1])

        scar_mask = np.linalg.norm(self.c - scar_point, axis=1) < scar_radius
        G_i[scar_mask] *= scar_factor
        G_e[scar_mask] *= scar_factor

        scar_mask_sim = np.linalg.norm(self.c_sim - scar_point, axis=1) < scar_radius
        G_i_sim[scar_mask_sim] *= scar_factor
        G_e_sim[scar_mask_sim] *= scar_factor

        return G_i, G_e, G_i_sim, G_e_sim, scar_center, scar_radius

    def gen_dataset(self, n_plot=0, indices=None):
        if indices is None:
            indices = range(self.data_nb)
        for i in tqdm(indices):
            out_path = Path(f"data/3D/data_functions/heart_potential_{i}.npz")
            if out_path.exists() and not (i < n_plot):
                continue
            self.config["plot"] = i < n_plot
            _, dt = self.gen_sample(i)
            y = self.A_obs @ self.epi_potential_fine
            # u: reconstruction-problem ground truth, on the closed sparse epicardium mesh
            u = self.P_fine_to_sparse @ self.epi_potential_fine
            np.savez(
                f"data/3D/data_functions/heart_potential_{i}.npz",
                u=u, dt=dt, y=y, u_fine=self.epi_potential_fine,
            )

            if self.config["plot"]:
                self.plot_potential_snapshots(
                    self.sparse_points, self.sparse_tris, u, None,
                    f"data/3D/plots/extracellular_potential_sparse_{i}.png",
                    vmin=u.min(), vmax=u.max(),
                )

                from meshio.xdmf import TimeSeriesWriter
                xdmf_dir = Path("data/3D/plots/xdmf")
                xdmf_dir.mkdir(parents=True, exist_ok=True)
                times = np.arange(u.shape[1]) * dt
                with contextlib.chdir(xdmf_dir):
                    with TimeSeriesWriter(f"epi_potential_sparse_{i}.xdmf") as writer:
                        writer.write_points_cells(self.sparse_points, [("triangle", self.sparse_tris)])
                        for k in range(u.shape[1]):
                            writer.write_data(float(times[k]), point_data={"u_epi_sparse": u[:, k].astype(np.float32)})

    def gen_fixed_data(self):
        """Generate fixed FEM operators and matrices"""
        out_dir = Path("data/3D/data_fixed")
        out_dir.mkdir(parents=True, exist_ok=True)

        fixed_data_path = out_dir / "fixed_data.npz"
        if fixed_data_path.exists():
            print("[gen_fixed_data] fixed_data.npz already exists on disk, skipping recompute", flush=True)
            self.build_observation_operator()
            return
        
        # Inverse-problem discretization on the sparse (closed) epicardium mesh
        mass_matrix = mass.assemble(self.sparse_surf_basis)[self.sparse_inds][:, self.sparse_inds].tocsc()
        mass_chol = cho_factor(mass_matrix)
        mass_matrix_inv = mass_chol.inv()
        dx = self.sparse_surf_basis.dx

        # Compute spatial gradient operator Ks
        grad_op = assemble_quadr_grad(self.sparse_surf_basis)
        grad_ops = [grad_i[:, self.sparse_inds] for grad_i in grad_op]
        proj_op = assemble_facet_proj_op(self.sparse_surf_basis)
        proj_ops = [proj_i for proj_i in proj_op]
        proj_grad_op_full = [(proj_q_op @ grad_q_op) for proj_q_op, grad_q_op in zip(proj_ops, grad_ops)]
        Ks = proj_grad_op_full[0]

        # Compute L2 projection from P_0 to P_1
        proj_p1 = mass_matrix_inv @ build_p0_to_p1_space(self.sparse_surf_basis, self.sparse_inds)

        # Inverse-problem forward operator
        cond_coarse_sparse = self.sigma_torso * np.repeat(np.eye(3)[None], self.coarse_sparse_epi.t.shape[1], axis=0)
        patch_node_inds, quad_matrix_elecs = assemble_electrode_patches_3d(
            self.coarse_sparse_pts, self.coarse_sparse_epi.boundary_facets(), self.coarse_sparse_epi.facets, self.elec_inds_coarse_sparse
        )
        transfer_op = assemble_transfer_op(
            self.coarse_sparse_epi, patch_node_inds, self.epi_inds_sparse_in_coarse,
            cond_coarse_sparse, self.d, None,
        )
        A = quad_matrix_elecs @ transfer_op

        L_data_fid = np.linalg.eigvalsh(mass_matrix_inv @ A.T @ A)[-1]

        np.savez_compressed(
            out_dir / "fixed_data.npz",
            M=mass_matrix.todense(),
            M_inv=mass_matrix_inv.todense(),
            dx=dx,
            Ks=Ks.todense(),
            A=A,
            proj_p1=proj_p1.todense(),
            L_data_fid=L_data_fid,
        )

        self.build_observation_operator()

    def build_observation_operator(self):
        """Assemble the electrode forward operator for the fine (full epi_endo) discretization"""
        out_dir = Path("data/3D/data_fixed")
        obs_path = out_dir / "fixed_data_obs.npz"
        interp_path = out_dir / "fixed_data_interp.npz"

        have_obs = obs_path.exists()
        have_interp = interp_path.exists()
        if have_obs:
            self.A_obs = np.load(obs_path)["A_obs"]
        if have_interp:
            self.P_fine_to_sparse = load_npz(interp_path)
        if have_obs and have_interp:
            print("[build_observation_operator] loading cached A_obs/P_fine_to_sparse from disk", flush=True)
            return

        out_dir.mkdir(parents=True, exist_ok=True)

        if not have_obs:
            surf_inds_global = self.local_to_global[self.surf_inds]
            heart_points_native = np.unique(self.tets[self.big_mesh.subdomains["heart"]])

            big_boundary_facets = self.big_mesh.boundary_facets()
            patch_node_inds, quad_matrix_elecs = assemble_electrode_patches_3d(self.points, big_boundary_facets, self.big_mesh.facets, self.elec_inds)

            cond_native = self.sigma_torso * np.repeat(np.eye(3)[None], self.tets.shape[0], axis=0)
            transfer_op_full = assemble_transfer_op(self.big_mesh, patch_node_inds, surf_inds_global, cond_native, self.d, heart_points_native)
            A_obs = quad_matrix_elecs @ transfer_op_full
            np.savez_compressed(obs_path, A_obs=A_obs)
            self.A_obs = A_obs

        if have_interp:
            return

        tet_idx, tet_bary, missing_mask = locate_in_tets(self.sparse_points, self.points, self.tets)

        # Only the torso nodes actually needed for these tets' interpolation weights
        necessary_torso_inds = np.unique(self.tets[tet_idx])
        local_remap = -np.ones(self.points.shape[0], dtype=np.int64)
        local_remap[necessary_torso_inds] = np.arange(necessary_torso_inds.size)
        tet_nodes_local = local_remap[self.tets[tet_idx]]  # (n_sparse, 4), local to necessary_torso_inds

        surf_inds_global = self.local_to_global[self.surf_inds]
        heart_points_native = np.unique(self.tets[self.big_mesh.subdomains["heart"]])
        heart_points_to_condense = np.setdiff1d(heart_points_native, necessary_torso_inds)
        cond_native = self.sigma_torso * np.repeat(np.eye(3)[None], self.tets.shape[0], axis=0)
        transfer_to_torso_nodes = assemble_transfer_op(self.big_mesh, necessary_torso_inds, surf_inds_global, cond_native, self.d, heart_points_to_condense,) 

        n_sparse = self.sparse_points.shape[0]
        interp_matrix = build_interp_matrix(np.arange(n_sparse), tet_bary, tet_nodes_local, n_sparse, necessary_torso_inds.size,)
        self.P_fine_to_sparse = csr_matrix(interp_matrix @ transfer_to_torso_nodes)
        save_npz(interp_path, self.P_fine_to_sparse)

    def gen_data_base_methods(self):
        """Generate fixed FEM operators for baseline methods, on the same sparse epicardium discretization as u / A (see build_sparse_mesh, gen_fixed_data)."""
        int_op_space = assemble_interpol_op_space(self.sparse_surf_basis)
        proj_elem_to_dof = build_p0_to_p1_space(self.sparse_surf_basis, self.sparse_inds)

        np.savez_compressed(
            "data/3D/data_fixed/fixed_data_base.npz",
            int_op_space=int_op_space.todense(),
            proj_elem_to_dof=proj_elem_to_dof.todense(),
        )

    def gen_csv(self):
        """Generate csv files for data loader"""
        if not os.path.exists("data/3D/data_csv"):
            os.makedirs("data/3D/data_csv")

        path = Path("data/3D/data_functions")
        path_save = Path("data/3D/data_csv")
        path_save.mkdir(parents=True, exist_ok=True)

        all_files = [f.name for f in path.iterdir() if f.is_file()]
        np.random.shuffle(all_files)

        train_frac, test_frac, val_frac = 0.8, 0.1, 0.1
        n_total = len(all_files)
        n_train = int(train_frac * n_total)
        n_test = int(test_frac * n_total)

        train_files = all_files[:n_train]
        test_files = all_files[n_train:n_train + n_test]
        val_files = all_files[n_train + n_test:]

        np.savetxt(path_save / "train.csv", train_files, fmt="%s", delimiter=", ")
        np.savetxt(path_save / "test.csv", test_files, fmt="%s", delimiter=", ")
        np.savetxt(path_save / "val.csv", val_files, fmt="%s", delimiter=", ")


def main():
    np.random.seed(42)
    with open("data_generation/config_data_3D.json") as f:
        config = json.load(f)

    n_plot_samples = config.get("n_plot_samples", 0)
    if config["plot"] or n_plot_samples > 0:
        if not os.path.exists("data/3D/plots"):
            os.makedirs("data/3D/plots")

    if not os.path.exists("data/3D/data_functions"):
        os.makedirs("data/3D/data_functions")

    gen = GenData3D(config)

    gen.gen_fixed_data()
    gen.gen_dataset(n_plot=n_plot_samples)
    gen.gen_csv()
    gen.gen_data_base_methods()
    compute_normalization_stats("3D")


if __name__ == "__main__":
    main()
