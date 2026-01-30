import json
import numpy as np
import meshio
import pyvista as pv
import matplotlib.pyplot as plt

from tqdm import trange
from pathlib import Path
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from sksparse.cholmod import cholesky

from skfem import ElementTriP1, FacetBasis, Basis

from utils_data import (
    convert_mesh,
    convert_mesh_pts_tris,
    normalize_points_minmax,
    mass,
    assemble_stiffness,
    assemble_facet_proj_op,
    assemble_quadr_grad,
    assemble_transfer_op,
    quadrature_matrix_all_electrodes,
    build_p0_to_p1_space,
    assemble_interpol_op_space
)


class GenData:
    """
    Data generator for 2D cardiac electrophysiology simulations
    using a monodomain-type model with Nagumo ionic dynamics.
    """

    def __init__(self, config):
        self.iteration = 0
        self.config = config
        
        # Dataset parameters
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

        # Load fine 2D heart mesh
        msh_fine = meshio.read("data/meshes/heart2d_fine.msh")
        self.pts_fine = msh_fine.points[:, :2]  # node coordinates
        self.elm_fine = msh_fine.cells_dict["triangle"]  # triangular elements
        heart_fine = convert_mesh_pts_tris(self.pts_fine, self.elm_fine)
        self.heart_basis_fine = Basis(
            heart_fine,
            ElementTriP1(),
        )
        self.dofs_per_elem = self.heart_basis_fine.elem.doflocs.shape[0]
        self.d = self.heart_basis_fine.mesh.p.shape[0]

        # Compute barycenter of each triangle
        self.c = self.pts_fine[self.elm_fine].mean(axis=1)

        # Fiber directions (circumferential)
        r = np.hypot(self.c[:, 0], self.c[:, 1])
        self.l = np.c_[self.c[:, 1] / r, -self.c[:, 0] / r]

        # Identity tensor per element
        self.I = np.repeat(np.eye(2)[None, :, :], self.c.shape[0], axis=0)

        # Fine-to-coarse mapping for epicardial surface
        print(f'Fine-to-coarse')
        self.torso = pv.UnstructuredGrid("data/meshes/torso2d.vtu")
        self.torso["pt_ids"] = np.arange(self.torso.n_points)
        pts_coarse = self.torso.threshold((2, 3), scalars="gmsh:geometrical").points[:,:2]
        heart_pv = self.torso.threshold((2, 3), scalars="gmsh:geometrical")
        heart = convert_mesh(heart_pv)
        
        self.pts_coarse_inds = heart_pv.point_data["pt_ids"]
        self.cond = self.torso.cell_data['G'].reshape(-1,2,2)
        self.torso_msh = convert_mesh(self.torso)
        self.elec_inds = np.loadtxt("data/meshes/elec_inds.txt", dtype=int).reshape(-1,3)

        # Define basis on heart surface facets
        self.heart_surf_basis = FacetBasis(
            heart,
            ElementTriP1(),
            facets=heart.boundary_facets(),
        )
        
        self.epi_inds = self.heart_surf_basis.get_dofs().all()

        # Normalize coordinates for nearest-neighbor mapping
        pts_fine_n = normalize_points_minmax(self.pts_fine)
        pts_coarse_n = normalize_points_minmax(pts_coarse)

        # KDTree for fast mapping from coarse to fine mesh
        tree = cKDTree(pts_fine_n.T)
        _, self.indices = tree.query(pts_coarse_n.T)
        
        if self.config["plot"]:
            from matplotlib.colors import ListedColormap
            from scipy.interpolate import interp1d
            import pandas as pd

            lut_vals = 1024
            rgb = pd.read_csv("/home/haas/cardiac/inverse_problem_ECGi/src/demo_2D/colormap/coolwarm_extended.csv")
            rgba = np.concatenate([rgb.to_numpy()/255, np.ones([rgb.shape[0], 1])], axis=1)
            rgba_interp = interp1d(np.linspace(0, 1, num=rgba.shape[0]), rgba.T)(np.linspace(0, 1, num=lut_vals)).T
            self.cmap_new = ListedColormap(rgba_interp)

    def fion(self, vm):
        """Nagumo ionic current model"""
        return (vm - self.Vdep)*(vm - self.Vrest)*(vm - self.Vthre)*self.gmax

    def gen_sample(self, seed):
        """Generate one simulation sample of cardiac potentials"""
        np.random.seed(seed)
        
        # Conductivities (intra- and extracellular)
        lamb_LT = np.random.uniform(self.config["bidomain_cond"]["lamb_LT"][0], self.config["bidomain_cond"]["lamb_LT"][1])
        eps = np.random.uniform(self.config["bidomain_cond"]["eps"][0], self.config["bidomain_cond"]["eps"][1])
        alpha = np.random.uniform(self.config["bidomain_cond"]["alpha"][0], self.config["bidomain_cond"]["alpha"][1]) 

        sigma_il = np.random.uniform(self.config["bidomain_cond"]["sigma_il"][0], self.config["bidomain_cond"]["sigma_il"][1]) 
        
        sigma_it = sigma_il*(1/lamb_LT)**2*((1+alpha*(1-eps))/(1+alpha))
        sigma_el = sigma_il*1/alpha
        sigma_et = sigma_it*1/(alpha*(1-eps))

        # Intracellular conductivity tensor
        G_i = sigma_it * self.I + (sigma_il - sigma_it) * self.l[:, :, None] @ self.l[:, None, :]

        # Extracellular conductivity tensor
        G_e = sigma_et * self.I + (sigma_el - sigma_et) * self.l[:, :, None] @ self.l[:, None, :]      
        
        # Random stimulation site
        stim_center = np.random.randint(0, self.pts_fine.shape[0])
        Istim = (np.linalg.norm(self.pts_fine - self.pts_fine[stim_center], axis=1) < 0.1).astype(float)
        
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
        M = mass.assemble(self.heart_basis_fine)
        K = assemble_stiffness(self.heart_basis_fine, G_m, self.d, self.dofs_per_elem)
        K_i = assemble_stiffness(self.heart_basis_fine, G_i, self.d, self.dofs_per_elem)
        K_e = assemble_stiffness(self.heart_basis_fine, G_e, self.d, self.dofs_per_elem)

        # Extracellular solver with small regularization for stability (pseudo-bidomain formulation)
        epsilon = 1e-9
        A_e = (K_i + K_e + epsilon * M).tocsc()
        solver_e = splu(A_e)
        
        time_sample = np.random.randint(
            self.sample_range[0], self.sample_range[1]
        )

        dt = np.random.uniform(self.dt_range[0], self.dt_range[1])
        ndt = int(np.rint(self.Tend / dt)) + 1
        
        A = (M * self.Cm + K * dt / self.beta).tocsc()
        solver = splu(A)

        # Initial condition: resting potential
        u = np.full(self.pts_fine.shape[0], self.Vrest)
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

        # Extracellular potential
        rhs_e = (-K_i @ u_hist)
        rhs_e = np.asarray(rhs_e)   # dense, fastest
        extra_fine = solver_e.solve(rhs_e)

        # Map to coarse epicardial mesh
        extra_coarse = extra_fine[self.indices]
        
        if self.config["plot"]:
            import matplotlib.colors as mcolors
            import matplotlib.cm as cm
            # Plot extracellular on fine mycardium over time
            tt = np.linspace(0, extra_fine.shape[1] - 1, 9).astype(int)

            # Global min and max for consistent color scale
            vmin = extra_fine.min()
            vmax = extra_fine.max()

            fig, axs = plt.subplots(1, 5, figsize=(16, 3))

            for ax, i in zip(axs.ravel(), tt[2:-2]):
                ax.tricontourf(
                    self.pts_fine[:, 0], self.pts_fine[:, 1], self.elm_fine,
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
            plt.savefig(f"data/plots/extracellular_potential_{seed}.png", dpi=300)
            plt.close(fig)

            
            # from matplotlib.colors import ListedColormap, BoundaryNorm
            # from scipy.interpolate import interp1d
            # import matplotlib.pyplot as plt
            # import pandas as pd
            # from matplotlib.cm import ScalarMappable

            # # --- Colormap ---
            # lut_vals = 1024
            # rgb = pd.read_csv("/home/haas/cardiac/inverse_problem_ECGi/src/demo_2D/colormap/coolwarm_extended.csv")
            # rgba = np.concatenate([rgb.to_numpy()/255, np.ones([rgb.shape[0], 1])], axis=1)
            # rgba_interp = interp1d(np.linspace(0, 1, num=rgba.shape[0]), rgba.T)(np.linspace(0, 1, num=lut_vals)).T
            # cmap_new = ListedColormap(rgba_interp)

            # # --- Plot setup ---
            # tt = np.linspace(0, extra_fine.shape[1] - 1, 7).astype(int)
            # tt_in = tt[1:-1]  # pick middle ones

            # vmin, vmax = extra_fine.min(), extra_fine.max()
            # n_levels = 24
            # levels = np.linspace(vmin, vmax, n_levels + 1)
            # norm = BoundaryNorm(levels, ncolors=cmap_new.N, clip=True)

            # # --- Use GridSpec for axes + colorbar ---
            # fig = plt.figure(figsize=(16, 3), dpi=300)
            # gs = fig.add_gridspec(1, 6, width_ratios=[1,1,1,1,1,0.05], wspace=0.1)  # last column for colorbar

            # axs = [fig.add_subplot(gs[0, i]) for i in range(5)]  # 5 plots
            # cbar_ax = fig.add_subplot(gs[0, 5])  # colorbar

            # # --- Plot each subplot ---
            # for ax, i in zip(axs, tt_in):
            #     t_val = i * time_sample * dt
            #     ax.tricontourf(self.pts_fine[:, 0], self.pts_fine[:, 1], self.elm_fine, extra_fine[:, i],
            #                 cmap=cmap_new, norm=norm, levels=levels, fraction=0.08)
            #     ax.set_title(f"t = {t_val:.1f} ms", fontsize=10)
            #     ax.axis('off')

            # # --- Colorbar ---
            # sm = ScalarMappable(cmap=cmap_new, norm=norm)
            # sm.set_array([])
            # fig.colorbar(sm, cax=cbar_ax, label='Extracellular potential [mV]', ticks=levels[::4])

            # plt.tight_layout()
            # plt.savefig(f"data/plots/extracellular_potential_{seed}.png")
            # plt.close(fig)


            
            # from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
            # import matplotlib.pyplot as plt
            # from scipy.interpolate import interp1d
            # import pandas as pd

            # lut_vals = 1024
            # rgb = pd.read_csv("/home/haas/cardiac/inverse_problem_ECGi/src/demo_2D/colormap/coolwarm_extended.csv")
            # rgba = np.concatenate([rgb.to_numpy()/255, np.ones([rgb.shape[0], 1])], axis=1)
            # rgba_interp = interp1d(np.linspace(0, 1, num=rgba.shape[0]), rgba.T)(np.linspace(0, 1, num=lut_vals)).T
            # cmap_new = ListedColormap(rgba_interp)
            
            # # select 2nd, 5th, and 8th from the linspace
            # # u_hist_norm = (extra_fine - extra_fine.min())/ (extra_fine.max() - extra_fine.min())
            
            # tt = [40, 100, 160]  # numpy indexing: 0-based
            # fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=300)  # 1 row, 3 columns
            # for ax, i in zip(axs.ravel(), tt):
            #     print(f"Plotting frame index: {i}")
            #     if i == 40:  # scar plot
            #         scar_vals = np.zeros(len(self.elm_fine))
            #         scar_vals[scar_mask_1] = 1
            #         scar_vals[scar_mask_2] = 1
            #         scar_cmap = ListedColormap(['lightgray', '#d62728'])
            #         ax.tripcolor(
            #             self.pts_fine[:, 0],
            #             self.pts_fine[:, 1],
            #             self.elm_fine,
            #             facecolors=scar_vals,
            #             cmap=scar_cmap
            #         )

            #     else:
            #         # normal voltage plot
            #         ax.tricontourf(
            #             self.pts_fine[:, 0],
            #             self.pts_fine[:, 1],
            #             self.elm_fine,
            #             extra_fine[:, i],
            #             vmin=extra_fine.min(),
            #             vmax=extra_fine.max(),
            #             cmap=cmap_new,
            #             levels=24
            #         )

            #     ax.axis('off')
            # # fig.colorbar(contour, ax=axs.ravel(), orientation='vertical', label='Voltage [V]')
            # plt.tight_layout()
            # plt.savefig(f"data/plots/heartbeat_with_scar_{self.iteration}.png", dpi=300)
            # plt.close(fig)
            # np.save(f"data/plots/heartbeat_with_scar_{self.iteration}.npy", extra_coarse)
        
        return extra_coarse[self.epi_inds], time_sample*dt
    
    def gen_scar(self, stim_center, G_i, G_e, scar_center_in=None, scar_radius_in=None):
        """Generate scar tissue with reduced conductivity"""
        scar_radius = np.random.uniform(self.config["scar"]["rad"][0], self.config["scar"]["rad"][1]) 
        while True:
            scar_center = np.random.randint(0, self.c.shape[0])
            # Check distance between scar and other scars/stimulation site to guarantee activation
            dist = np.linalg.norm(self.c[scar_center] - self.pts_fine[stim_center])
            if scar_center_in is not None:
                dist_scar = np.linalg.norm(self.c[scar_center] - self.c[scar_center_in])
                check_scar_dist = dist_scar > scar_radius_in + 0.1
            else:
                check_scar_dist = True  
            if dist > scar_radius + 0.1 and check_scar_dist:
                break

        scar_mask = np.linalg.norm(self.c - self.c[scar_center], axis=1) < scar_radius
        scar_factor = np.random.uniform(self.config["scar"]["cond_factor"][0], self.config["scar"]["cond_factor"][1]) 
        G_i[scar_mask] *= scar_factor
        G_e[scar_mask] *= scar_factor
        return G_i, G_e, scar_center, scar_radius
        

    def gen_dataset(self):
        """Generate entire dataset and save"""
        for i in trange(self.data_nb):
            u, dt = self.gen_sample(i)
            np.savez(f"data/data_functions/heart_potential_{i}.npz", u=u, dt=dt)
            
    def gen_fixed_data(self):
        """Generate fixed FEM operators and matrices"""
        
        # Compute spatial mass matrix, inverse, and quadrature weights
        mass_matrix = mass.assemble(self.heart_surf_basis)[self.epi_inds][:, self.epi_inds].tocsc()
        mass_chol = cholesky(mass_matrix)
        mass_matrix_inv = mass_chol.inv()
        dx = self.heart_surf_basis.dx
        
        # Compute spatial gradient operator Ks
        grad_op = assemble_quadr_grad(self.heart_surf_basis)
        grad_ops = [grad_i[:, self.epi_inds] for grad_i in grad_op] 
        proj_op = assemble_facet_proj_op(self.heart_surf_basis)
        proj_ops = [proj_i for proj_i in proj_op] 
        proj_grad_op_full = [(proj_q_op @ grad_q_op) for proj_q_op, grad_q_op in zip(proj_ops, grad_ops)]
        Ks = proj_grad_op_full[0]
        
        # Solve forwad problem in ECGI and compute the forward operator A as well as \tilde{A} integrated over space
        d=2
        transfer_op = assemble_transfer_op(self.torso_msh, self.elec_inds, self.epi_inds, self.cond, d, self.pts_coarse_inds)
        quad_matrix_elecs = quadrature_matrix_all_electrodes(self.elec_inds, self.torso)
        A = quad_matrix_elecs@transfer_op
        
        # Compute L2 projection from P_0 to P_1
        proj_p1 = mass_matrix_inv@build_p0_to_p1_space(self.heart_surf_basis, self.epi_inds)
        
        # Compute Lipschitz constant of data fidelity function G
        L_data_fid = np.linalg.eigvalsh(mass_matrix_inv@A.T@A)[-1] 
        
        np.savez_compressed(
            "data/data_fixed/fixed_data.npz",
            M=mass_matrix.todense(),
            M_inv=mass_matrix_inv.todense(),
            dx=dx,
            Ks=Ks.todense(),
            A=A,
            proj_p1=proj_p1.todense(),
            L_data_fid=L_data_fid
        )
        
    def gen_data_base_methods(self):
        """Generate fixed FEM operators for baseline methods"""
        
        # Compute interpolation operator to map function values of spatial gradient to nodes
        int_op_space = assemble_interpol_op_space(self.heart_surf_basis)
        
        # Compute interpolation from elementwise to nodewise dofs
        proj_elem_to_dof = build_p0_to_p1_space(self.heart_surf_basis, self.epi_inds)
        
        np.savez_compressed("data/data_fixed/fixed_data_base.npz", int_op_space=int_op_space.todense(), proj_elem_to_dof=proj_elem_to_dof.todense())

        
    def gen_csv(self):
        path = Path("data/data_functions")
        path_save = Path("data/data_csv")
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
    with open("data_generation/config_data.json") as f:
        config = json.load(f)

    gen = GenData(config)
    
    gen.gen_dataset()
    gen.gen_fixed_data()
    gen.gen_csv()
    
    gen.gen_data_base_methods()


if __name__ == "__main__":
    main()