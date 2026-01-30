import torch

def L2(u, v, M, D, Ks=None, Kt=None, dx=None, dt=None):
    diff = u - v
    tmp = torch.einsum('...ij,...ki->...kj', diff, M)
    tmp = torch.einsum('...ij,...kj->...ik', tmp, D)
    return torch.sqrt(torch.sum(diff * tmp))

def L2_squared(u, v, M, D):
    diff = u - v
    tmp = torch.einsum('...ij,...ki->...kj', diff, M)
    tmp = torch.einsum('...ij,...kj->...ik', tmp, D)
    return torch.sum(diff * tmp)

def L2_squared_torso(u, z, A, D):
    diff = A@u - z
    tmp = torch.einsum('...ij,...kj->...ik', diff, D)
    return torch.sum(diff * tmp)

def H1(u, v, M, D, Ks, Kt, dx, dt):
    diff = u - v
    
    tmp = torch.einsum('...ij,...ki->...kj', diff, M)
    tmp = torch.einsum('...ij,...kj->...ik', tmp, D)
    tmp_comp = torch.sum(diff * tmp)
    
    grad_s = torch.einsum('...ij,...ki->...kj', diff, Ks)
    grad_s_weight = dx.flatten().view(1, 1, -1, 1)*torch.einsum('...ij,...kj->...ik', grad_s, D)
    grad_s_comb = torch.sum(grad_s*grad_s_weight)
    
    grad_t = torch.einsum('...ij,...kj->...ik', diff, Kt)
    grad_t_weight = dt *torch.einsum('...ij,...ki->...kj', grad_t, M)
    grad_t_comb = torch.sum(grad_t*grad_t_weight)
    
    return torch.sqrt(tmp_comp + grad_s_comb + grad_t_comb)


def AGDR(x_init, y, proj_p1, Ks, M, M_inv, D, D_inv, dt, A, L_data_fid, model, sigma, max_iter=300, tol=1e-4):
    """
    Optimization using the FISTA accelerated rule
    """

    # initial value: noisy image
    x = x_init.clone()
    z = x.clone()
    t = torch.ones(x.shape[0], device=x.device).view(-1, 1, 1, 1)

    # cache values of scaling coeff for efficiency
    scaling = model.get_scaling(sigma=sigma)

    # the index of the images that have not converged yet
    idx = torch.arange(0, x.shape[0], device=x.device)
    # relative change in the estimate
    res = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    old_cost = 1e12*torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

    # mean number of iterations over the batch
    i_mean = 0
    for i in range(max_iter):
        model.scaling = scaling[idx]
        x_old = torch.clone(x)
        grad, cost = model.reconstruct(z[idx], y[idx], sigma[idx], proj_p1, Ks, M, M_inv, D[idx], D_inv[idx], dt[idx],A, L_data_fid)
        x[idx] = z[idx] - grad

        t_old = torch.clone(t)
        t = 0.5 * (1 + torch.sqrt(1 + 4*t**2))
        z[idx] = x[idx] + (t_old[idx] - 1)/t[idx] * (x[idx] - x_old[idx])

        if i > 10:
            res[idx] = L2(x[idx], x_old[idx], M, D[idx])/L2(x[idx], torch.zeros_like(x[idx]), M, D[idx]) 

        esti = cost - old_cost[idx]
        old_cost[idx] = cost
        id_restart = (esti > 0).nonzero().view(-1)
        t[idx[id_restart]] = 1
        z[idx[id_restart]] = x[idx[id_restart]]

        condition = (res > tol)
        idx = condition.nonzero().view(-1)
        i_mean += torch.sum(condition).item() / x.shape[0]

        if torch.max(res) < tol:
            break

    model.clear_cache()

    return x, i+1, i_mean+1


def proj_l1_channel(x):
    """
    Projects a batch of vectors x in dimension d onto the unit l1-ball.
    The dimensions d and K are defined in the paper, H and W are the height and width of the image.

    Args:
        x torch.Tensor: Input tensor of shape (batch, d, K, H, W).
    """
    norm_x = torch.norm(x, p=1, dim=1, keepdim=True)
    mask = (norm_x <= 1.0).repeat(1, x.shape[1], 1, 1, 1)

    if mask.all():
        return x  # Already within the l1-ball

    abs_x = torch.abs(x)
    sorted_x, _ = torch.sort(abs_x, descending=True, dim=1)
    cumsum_x = torch.cumsum(sorted_x, dim=1)

    input_dim = x.shape[1]
    rho = (sorted_x * torch.arange(1, input_dim + 1, device=x.device)
           [None, :, None, None, None] > (cumsum_x - 1.0)).sum(dim=1) - 1

    # This is necessary due to numerical errors
    rho[rho < 0] = 0

    theta = (torch.gather(cumsum_x, 1, rho.unsqueeze(1)) - 1.0) / \
        (rho + 1).unsqueeze(1)
    projected_x = torch.sign(x) * torch.clamp(abs_x - theta, min=0)
    projected_x[mask] = x[mask]

    return projected_x