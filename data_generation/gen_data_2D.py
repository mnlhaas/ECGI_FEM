import json
import os
import meshio
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from tqdm import trange
from pathlib import Path
from sksparse.cholmod import cho_factor

from skfem import ElementTriP1, FacetBasis, Basis, MeshTri

from utils_data import (
    convert_mesh,
    convert_mesh_pts_tris,
    mass,
    assemble_stiffness,
    assemble_facet_proj_op,
    assemble_quadr_grad,
    assemble_transfer_op,
    quadrature_matrix_all_electrodes,
    build_p0_to_p1_space,
    assemble_interpol_op_space,
    compute_normalization_stats,
    locate_on_triangles,
    build_interp_matrix,
)


class GenData2D:
    """
    Data generator for 2D cardiac electrophysiology simulations
    using a monodomain-type model with Nagumo ionic dynamics.
    """

    def __init__(self, config):
        self.iteration = 0
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

        self.d = 2

        self.heart_center = np.array([20.0, -40.0])  # local heart center, torso mm coords
        self.torso = pv.UnstructuredGrid("data/meshes/2D/torso2d.vtu")
        self.torso["pt_ids"] = np.arange(self.torso.n_points)
        self.points = self.torso.points[:, :2]
        self.tris = self.torso.cells_dict[pv.CellType.TRIANGLE]
        self.cond = self.torso.cell_data["G"].reshape(-1, 2, 2)
        region_tags = self.torso.cell_data["gmsh:geometrical"]
        heart_elem_idx = np.where(region_tags == 2)[0]  # myocardium only, for EP simulation

        self.big_mesh = MeshTri(self.points.T, self.tris.T, _subdomains={"heart": heart_elem_idx})
        self.torso_msh = self.big_mesh 

        # Heart-only coarse mesh
        heart_pv = self.torso.threshold((2, 3), scalars="gmsh:geometrical")
        self.pts_coarse_inds = heart_pv.point_data["pt_ids"]
        self.local_to_global = self.pts_coarse_inds
        heart = convert_mesh(heart_pv)

        self.elec_inds = np.loadtxt("data/meshes/2D/elec_inds.txt", dtype=int).reshape(-1, 3)

        self.heart_surf_basis = FacetBasis(heart, ElementTriP1(), facets=heart.boundary_facets())
        self.epi_inds = self.heart_surf_basis.get_dofs().all()
        self.epi_inds_global = self.local_to_global[self.epi_inds]
        self.coarse_epi_pts_cm = (heart.p.T[self.epi_inds] - self.heart_center) * 0.1

        # Independently refined heart-only mesh 
        self.build_fine_heart_mesh()
        # Independently-meshed fine myocardium mesh actually used for the EP simulation
        self.build_simulation_mesh()
        # Interpolate simulation results (on the .msh mesh) onto the refined mesh's epicardial
        self.build_interpolation_operators()

        if self.config["plot"] or self.config.get("n_plot_samples", 0) > 0:
            from matplotlib.colors import ListedColormap
            from scipy.interpolate import interp1d
            import pandas as pd

            lut_vals = 1024
            rgb = pd.read_csv("data/colormap/coolwarm_extended.csv")
            rgba = np.concatenate([rgb.to_numpy() / 255, np.ones([rgb.shape[0], 1])], axis=1)
            rgba_interp = interp1d(np.linspace(0, 1, num=rgba.shape[0]), rgba.T)(np.linspace(0, 1, num=lut_vals)).T
            self.cmap_new = ListedColormap(rgba_interp)

    def build_fine_heart_mesh(self):
        """Build the independently-refined myocardium-only mesh used for the observation
        operator's geometry, and as an interpolation target (see build_interpolation_operators)."""
        self.fine = self.big_mesh.refined(self.config.get("refine_levels", 1))
        heart_elem_fine = self.fine.subdomains["heart"]
        heart_tris_fine_global = self.fine.t[:, heart_elem_fine].T  # (n_elem, 3), global fine-mesh indices
        self.heart_points_fine_global = np.unique(heart_tris_fine_global)

        heart_tris_fine_local = np.searchsorted(self.heart_points_fine_global, heart_tris_fine_global)
        heart_pts_fine_raw = self.fine.p.T[self.heart_points_fine_global]  # torso-native mm, global position
        self.heart_pts_fine = (heart_pts_fine_raw - self.heart_center) * 0.1  # cm, heart-centered
        self.heart_tris_fine = heart_tris_fine_local
        self.heart_fine = MeshTri(self.heart_pts_fine.T, heart_tris_fine_local.T)

        # Fine epicardium
        epi_fine_local_both = FacetBasis(self.heart_fine, ElementTriP1(), facets=self.heart_fine.boundary_facets()).get_dofs().all()
        r_epi_fine = np.linalg.norm(self.heart_pts_fine[epi_fine_local_both], axis=1)
        self.epi_inds_fine_local = epi_fine_local_both[r_epi_fine > 2.5]
        self.epi_inds_fine_global = self.heart_points_fine_global[self.epi_inds_fine_local]

    def build_simulation_mesh(self):
        """Load the independently-meshed fine myocardium mesh"""
        msh_fine = meshio.read("data/meshes/2D/heart2d_fine.msh")
        self.sim_pts = msh_fine.points[:, :2]
        self.sim_tris = msh_fine.cells_dict["triangle"]
        sim_mesh = convert_mesh_pts_tris(self.sim_pts, self.sim_tris)
        self.sim_basis = Basis(sim_mesh, ElementTriP1())
        self.dofs_per_elem_sim = self.sim_basis.elem.doflocs.shape[0]

        # Fiber directions (circumferential)
        self.c_sim = self.sim_pts[self.sim_tris].mean(axis=1)
        r_sim = np.hypot(self.c_sim[:, 0], self.c_sim[:, 1])
        self.l_sim = np.c_[self.c_sim[:, 1] / r_sim, -self.c_sim[:, 0] / r_sim]
        self.I_sim = np.repeat(np.eye(2)[None, :, :], self.c_sim.shape[0], axis=0)

    def build_interpolation_operators(self):
        """Interpolate simulation results"""
        def pad3(pts):
            return np.concatenate([pts, np.zeros((pts.shape[0], 1))], axis=1)

        n_sim = self.sim_pts.shape[0]
        sim_pts_3d = pad3(self.sim_pts)
        target_refined = self.heart_pts_fine[self.epi_inds_fine_local]
        target_sparse = self.coarse_epi_pts_cm

        tri_idx, bary, _ = locate_on_triangles(pad3(target_refined), sim_pts_3d, self.sim_tris)
        self.P_sim_to_refined_epi = build_interp_matrix(tri_idx, bary, self.sim_tris, target_refined.shape[0], n_sim)

        tri_idx, bary, _ = locate_on_triangles(pad3(target_sparse), sim_pts_3d, self.sim_tris)
        self.P_sim_to_sparse = build_interp_matrix(tri_idx, bary, self.sim_tris, target_sparse.shape[0], n_sim)

    def fion(self, vm):
        """Nagumo ionic current model"""
        return (vm - self.Vrest) * (vm - self.Vthre) * (vm - self.Vdep) * self.gmax

    def gen_sample(self, seed):
        """Generate one simulation sample of cardiac potentials"""
        np.random.seed(seed)

        # Conductivities (intra- and extracellular)
        lamb_LT = np.random.uniform(self.config["bidomain_cond"]["lamb_LT"][0], self.config["bidomain_cond"]["lamb_LT"][1])
        eps = np.random.uniform(self.config["bidomain_cond"]["eps"][0], self.config["bidomain_cond"]["eps"][1])

        alpha = self.config["bidomain_cond"]["alpha"]
        sigma_il = self.config["bidomain_cond"]["sigma_il"]

        sigma_it = sigma_il*(1/lamb_LT)**2*((1+alpha*(1-eps))/(1+alpha))
        sigma_el = sigma_il*1/alpha
        sigma_et = sigma_it*1/(alpha*(1-eps))

        # Intracellular conductivity tensor
        G_i = sigma_it * self.I_sim + (sigma_il - sigma_it) * self.l_sim[:, :, None] @ self.l_sim[:, None, :]

        # Extracellular conductivity tensor
        G_e = sigma_et * self.I_sim + (sigma_el - sigma_et) * self.l_sim[:, :, None] @ self.l_sim[:, None, :]

        # Random stimulation site
        stim_center = np.random.randint(0, self.sim_pts.shape[0])
        Istim = (np.linalg.norm(self.sim_pts - self.sim_pts[stim_center], axis=1) < 0.1).astype(float)

        # Generate scar tissue away from stimulation site
        if np.random.rand() < self.config["scar"]["prob"]:
            G_i, G_e, scar_center_in, scar_radius_in = self.gen_scar(stim_center, G_i, G_e)
            if np.random.rand() < self.config["scar"]["prob_second"]:
                G_i, G_e, _, _ = self.gen_scar(stim_center, G_i, G_e, scar_center_in, scar_radius_in)

        # Monodomain effective conductivity (G_m = G_i * (G_i+G_e)^-1 * G_e)
        G_sum = G_i + G_e
        G_sum_inv = np.linalg.inv(G_sum)
        G_m = np.einsum("nij,njk,nkl->nil", G_i, G_sum_inv, G_e)

        # Assemble FEM mass and stiffness matrices
        M = mass.assemble(self.sim_basis)
        K = assemble_stiffness(self.sim_basis, G_m, self.d, self.dofs_per_elem_sim)
        K_i = assemble_stiffness(self.sim_basis, G_i, self.d, self.dofs_per_elem_sim)
        K_e = assemble_stiffness(self.sim_basis, G_e, self.d, self.dofs_per_elem_sim)

        # Extracellular solver with small regularization for stability (pseudo-bidomain formulation)
        epsilon = 1e-9
        A_e = (K_i + K_e + epsilon * M).tocsc()
        solver_e = cho_factor(A_e)

        time_sample = np.random.randint(
            self.sample_range[0], self.sample_range[1]
        )

        dt = np.random.uniform(self.dt_range[0], self.dt_range[1])
        ndt = int(np.rint(self.Tend / dt)) + 1

        A = (M * self.Cm + K * dt / self.beta).tocsc()
        solver = cho_factor(A)

        # Initial condition: resting potential
        u = np.full(self.sim_pts.shape[0], self.Vrest)
        u_hist = [u.copy()]
        for i in range(1, ndt):
            t = i * dt

            # Ionic current and stimulation
            Iion = self.fion(u)
            Is = Istim * (self.Imax if t < self.Idur else 0.0)

            # Backward Euler step
            b = M @ (u * self.Cm - (Iion - Is) * dt)
            u = solver.solve(b)

            if i % time_sample == 0:
                u_hist.append(u.copy())

        u_hist = np.array(u_hist).T

        # Extracellular potential on the simulation mesh
        rhs_e = (-K_i @ u_hist)
        rhs_e = np.asarray(rhs_e)
        extra_fine = solver_e.solve(rhs_e)

        # Interpolate onto the sparse mesh's epicardial nodes
        extra_coarse = self.P_sim_to_sparse @ extra_fine

        self.epi_potential_fine = self.P_sim_to_refined_epi @ extra_fine

        if self.config["plot"]:
            import matplotlib.colors as mcolors
            import matplotlib.cm as cm
            # Plot extracellular on fine myocardium over time
            tt = np.linspace(0, extra_fine.shape[1] - 1, 9).astype(int)

            vmin = extra_fine.min()
            vmax = extra_fine.max()

            fig, axs = plt.subplots(1, 5, figsize=(16, 3))

            for ax, i in zip(axs.ravel(), tt[2:-2]):
                ax.tricontourf(
                    self.sim_pts[:, 0], self.sim_pts[:, 1], self.sim_tris,
                    extra_fine[:, i], cmap=self.cmap_new, vmin=vmin, vmax=vmax, levels=24
                )
                ax.axis('off')

            levels = np.linspace(vmin, vmax, 25)
            norm = mcolors.BoundaryNorm(levels, ncolors=self.cmap_new.N)
            sm = cm.ScalarMappable(cmap=self.cmap_new, norm=norm)
            sm.set_array([])
            fig.colorbar(
                sm,
                ax=axs,
                orientation='vertical',
                pad=0.02,
                fraction=0.05,
                norm=norm,
                label='Extracellular potential [mV]'
            )
            plt.savefig(f"data/2D/plots/extracellular_potential_{seed}.png", dpi=300)
            plt.close(fig)

        return extra_coarse, time_sample*dt

    def gen_scar(self, stim_center, G_i, G_e, scar_center_in=None, scar_radius_in=None):
        """Generate scar tissue with reduced conductivity, on the simulation mesh
        (the resolution gen_sample() actually simulates on)."""
        scar_radius = np.random.uniform(self.config["scar"]["rad"][0], self.config["scar"]["rad"][1])
        while True:
            scar_center = np.random.randint(0, self.c_sim.shape[0])
            # Check distance between scar and other scars/stimulation site to guarantee activation
            dist = np.linalg.norm(self.c_sim[scar_center] - self.sim_pts[stim_center])
            if scar_center_in is not None:
                dist_scar = np.linalg.norm(self.c_sim[scar_center] - self.c_sim[scar_center_in])
                check_scar_dist = dist_scar > scar_radius_in + 0.1
            else:
                check_scar_dist = True
            if dist > scar_radius + 0.1 and check_scar_dist:
                break

        scar_mask = np.linalg.norm(self.c_sim - self.c_sim[scar_center], axis=1) < scar_radius
        scar_factor = np.random.uniform(self.config["scar"]["cond_factor"][0], self.config["scar"]["cond_factor"][1])
        G_i[scar_mask] *= scar_factor
        G_e[scar_mask] *= scar_factor
        return G_i, G_e, scar_center, scar_radius

    def gen_dataset(self, n_plot=0):
        """Generate entire dataset"""
        for i in trange(self.data_nb):
            self.config['plot'] = i < n_plot
            u, dt = self.gen_sample(i)
            y = self.A_obs @ self.epi_potential_fine
            np.savez(
                f"data/2D/data_functions/heart_potential_{i}.npz",
                u=u, dt=dt, y=y, u_fine=self.epi_potential_fine,
            )

    def gen_fixed_data(self):
        """Generate fixed FEM operators and matrices"""
        if not os.path.exists("data/2D/data_fixed"):
            os.makedirs("data/2D/data_fixed")

        # Compute spatial mass matrix, inverse, and quadrature weights
        mass_matrix = mass.assemble(self.heart_surf_basis)[self.epi_inds][:, self.epi_inds].tocsc()
        mass_chol = cho_factor(mass_matrix)
        mass_matrix_inv = mass_chol.inv()
        dx = self.heart_surf_basis.dx

        # Compute spatial gradient operator Ks
        grad_op = assemble_quadr_grad(self.heart_surf_basis)
        grad_ops = [grad_i[:, self.epi_inds] for grad_i in grad_op]
        proj_op = assemble_facet_proj_op(self.heart_surf_basis)
        proj_ops = [proj_i for proj_i in proj_op]
        proj_grad_op_full = [(proj_q_op @ grad_q_op) for proj_q_op, grad_q_op in zip(proj_ops, grad_ops)]
        Ks = proj_grad_op_full[0]

        # Inverse-problem forward operator on the sparser mesh
        epi_inds_global = self.local_to_global[self.epi_inds]
        transfer_op = assemble_transfer_op(self.big_mesh, self.elec_inds, epi_inds_global, self.cond, self.d, self.local_to_global)
        quad_matrix_elecs = quadrature_matrix_all_electrodes(self.elec_inds, self.torso)
        A = quad_matrix_elecs @ transfer_op

        # Compute L2 projection from P_0 to P_1
        proj_p1 = mass_matrix_inv@build_p0_to_p1_space(self.heart_surf_basis, self.epi_inds)

        # Compute Lipschitz constant of data fidelity function G
        L_data_fid = np.linalg.eigvalsh(mass_matrix_inv@A.T@A)[-1]

        np.savez_compressed(
            "data/2D/data_fixed/fixed_data.npz",
            M=mass_matrix.todense(),
            M_inv=mass_matrix_inv.todense(),
            dx=dx,
            Ks=Ks.todense(),
            A=A,
            proj_p1=proj_p1.todense(),
            L_data_fid=L_data_fid
        )

        self.build_observation_operator()

    def build_observation_operator(self):
        """Assemble the electrode forward operator on the refined torso-heart mesh"""
        n_elem_fine = self.fine.t.shape[1]
        n_elem_coarse = self.tris.shape[0]
        assert self.config.get("refine_levels", 1) == 1, "cond_fine parent lookup assumes a single uniform refinement"
        cond_fine = self.cond[np.arange(n_elem_fine) % n_elem_coarse]

        transfer_op = assemble_transfer_op(self.fine, self.elec_inds, self.epi_inds_fine_global, cond_fine, self.d, self.heart_points_fine_global)
        quad_matrix_elecs = quadrature_matrix_all_electrodes(self.elec_inds, self.torso)
        A_obs = quad_matrix_elecs @ transfer_op

        np.savez_compressed("data/2D/data_fixed/fixed_data_obs.npz", A_obs=A_obs)
        self.A_obs = A_obs

    def gen_data_base_methods(self):
        """Generate fixed FEM operators for baseline methods"""
        # Compute interpolation operator to map function values of spatial gradient to nodes
        int_op_space = assemble_interpol_op_space(self.heart_surf_basis)

        # Compute interpolation from elementwise to nodewise dofs
        proj_elem_to_dof = build_p0_to_p1_space(self.heart_surf_basis, self.epi_inds)

        np.savez_compressed("data/2D/data_fixed/fixed_data_base.npz", int_op_space=int_op_space.todense(), proj_elem_to_dof=proj_elem_to_dof.todense())


    def gen_csv(self):
        """Generate csv files for data loader"""
        if not os.path.exists("data/2D/data_csv"):
            os.makedirs("data/2D/data_csv")

        path = Path("data/2D/data_functions")
        path_save = Path("data/2D/data_csv")
        path_save.mkdir(parents=True, exist_ok=True)

        all_files = [f.name for f in path.iterdir() if f.is_file()]
        np.random.shuffle(all_files)

        # Split fractions
        train_frac, test_frac, val_frac = 0.8, 0.1, 0.1
        n_total = len(all_files)
        n_train = int(train_frac * n_total)
        n_test = int(test_frac * n_total)

        train_files = all_files[:n_train]
        test_files = all_files[n_train:n_train+n_test]
        val_files = all_files[n_train+n_test:]

        np.savetxt(path_save / "train.csv", train_files, fmt="%s", delimiter=", ")
        np.savetxt(path_save / "test.csv", test_files, fmt="%s", delimiter=", ")
        np.savetxt(path_save / "val.csv", val_files, fmt="%s", delimiter=", ")




def main():
    np.random.seed(42)
    # Load simulation configuration
    with open("data_generation/config_data_2D.json") as f:
        config = json.load(f)

    if config["plot"]:
        if not os.path.exists("data/2D/plots"):
            os.makedirs("data/2D/plots")

    if not os.path.exists("data/2D/data_functions"):
            os.makedirs("data/2D/data_functions")

    gen = GenData2D(config)

    gen.gen_fixed_data()
    gen.gen_dataset(n_plot=config.get("n_plot_samples", 0))
    gen.gen_csv()
    gen.gen_data_base_methods()
    compute_normalization_stats("2D")


if __name__ == "__main__":
    main()
