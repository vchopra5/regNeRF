import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_intervals(num_rays, near=2.0, far=6.0, N=64, device=None, dtype=torch.float32):
    t_vals = torch.linspace(near, far, N+1, device=device, dtype=dtype)
    return t_vals.expand(num_rays, -1)


def conical_frustum_to_gaussian(rays_o, rays_d, t_vals, radii):

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    radii = radii.reshape(-1, 1)

    t0= t_vals[..., :-1]
    t1 = t_vals[..., 1:]

    t_mu= (t0 + t1) / 2
    t_delta = (t1 - t0)/ 2

    denom  = 3*(t_mu**2) + (t_delta**2)

    mean_t = t_mu + (2*t_mu*t_delta**2) / denom

    var_t = (
        t_delta**2 / 3
        - (4 / 15)
        * (t_delta**4 * (12 * t_mu**2 - t_delta**2))
        /denom**2
    )

    var_r = radii**2 * (
        t_mu**2 / 4
        + 5 * t_delta**2 / 12
        - (4/15)*t_delta**4/denom
    )

    means = rays_o[:, None, :] + rays_d[:, None, :]*mean_t[..., None]

    d_sq = rays_d**2
    d_mag_sq = d_sq.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    cov_diag = (
        var_t[..., None]*d_sq[:, None, :]
        + var_r[..., None]*(1 - d_sq / d_mag_sq)[:, None, :]
    )

    return means, cov_diag

# For the conical rays
def integrated_positional_encoding(means, cov_diag, min_deg=0, max_deg=16):

    scales = 2.0 ** torch.arange(min_deg, max_deg, device=means.device, dtype=means.dtype)

    scaled_means = means[..., None, :] * scales[None, None, :, None]
    scaled_vars = cov_diag[..., None, :]*scales[None, None, :, None] ** 2

    attenuation = torch.exp(-0.5*scaled_vars)

    sin_features = torch.sin(scaled_means) * attenuation
    cos_features = torch.cos(scaled_means) * attenuation

    sin_features = sin_features.flatten(start_dim=-2)
    cos_features = cos_features.flatten(start_dim=-2)

    return torch.cat([sin_features, cos_features], dim=-1)

# For viewing directions
def positional_encoding(x, num_freqs=4):
    scales = 2 ** (
        torch.arange(
            num_freqs, device=x.device, dtype=x.dtype
        )
    )

    scaled = x[..., None, :]*scales[:, None]

    return torch.cat([
        x,
        torch.sin(scaled).flatten(-2),
        torch.cos(scaled).flatten(-2)
    ], dim=-1)


def volume_render(rgb, density, t_vals, rays_d, white_bkgd=True):

    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = delta * torch.linalg.norm(
        rays_d, dim=-1, keepdim=True
    )

    sigma = density.squeeze(-1)
    alpha = 1.0 - torch.exp(-sigma*delta)

    transmittance = torch.cumprod(
        torch.cat([
            torch.ones_like(alpha[..., :1]),
            1.0 - alpha + 1e-10
        ], dim=-1),
        dim=-1
    )[..., :-1]

    weights = alpha * transmittance

    if rgb is None:
        return None, weights

    rgb_map = torch.sum(
        weights[..., None] * rgb,
        dim = -2
    )


    acc = weights.sum(dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1 - acc[..., None])

    return rgb_map, weights

def resample_intervals(t_vals, weights, N=64, randomized=True, generator=None):
    """
    t_vals:  [B, M+1] coarse interval boundaries
    weights: [B, M]   coarse rendering weights

    Returns: [B, N+1] new interval boundaries
    """

    # Smooth the weights so nearby regions are considered
    weights_pad = torch.cat([
        weights[..., :1],
        weights,
        weights[..., -1:]
    ], dim=-1)

    weights_max = torch.maximum(
        weights_pad[..., :-1],
        weights_pad[..., 1:]
    )

    weights_blur = 0.5 * (
        weights_max[..., :-1] + weights_max[..., 1:]
    )

    # Avoid zero-probability intervals
    weights_blur = weights_blur + 0.01

    # Normalize weights into probabilities
    pdf = weights_blur / weights_blur.sum(
        dim=-1, keepdim=True
    )

    # Build cumulative distribution
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([
        torch.zeros_like(cdf[..., :1]), cdf
    ], dim=-1)

    # Sample N+1 quantiles between 0 and 1
    if randomized:
        u = (
            torch.arange(
                N + 1, device=t_vals.device,
                dtype=t_vals.dtype
            )[None, :]
            + torch.rand(
                t_vals.shape[0], N + 1,
                device=t_vals.device, generator=generator,
                dtype=t_vals.dtype
            )
        ) / (N + 1)

        u[..., 0] = 0.0
        u[..., -1] = 1.0
    else:
        u = torch.linspace(
            0.0, 1.0, N + 1,
            device=t_vals.device,
            dtype=t_vals.dtype
        ).expand(t_vals.shape[0], -1)

    # Find which CDF interval contains each sample
    inds = torch.searchsorted(
        cdf.contiguous(), u.contiguous(), right=True
    )

    below = (inds - 1).clamp(min=0, max=cdf.shape[-1] - 1)
    above = inds.clamp(max=cdf.shape[-1] - 1)

    cdf_below = torch.gather(cdf, -1, below)
    cdf_above = torch.gather(cdf, -1, above)

    t_below = torch.gather(t_vals, -1, below)
    t_above = torch.gather(t_vals, -1, above)

    # Interpolate to get new depth boundaries
    denom = (cdf_above - cdf_below).clamp_min(1e-8)
    fraction = (u - cdf_below) / denom

    t_new = t_below + fraction * (t_above - t_below)

    return t_new.detach()

class MipNeRFMLP(nn.Module):
    def __init__(self, pos_dim=96, hidden_dim=256, depth=8):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(depth):
            in_dim = pos_dim if i == 0 else hidden_dim

            if i == 4:
                in_dim += pos_dim

            self.layers.append(nn.Linear(in_dim, hidden_dim))

        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        dir_dim = 3 + 4*3*2

        self.color_layer = nn.Linear(hidden_dim + dir_dim, 128)
        self.color_head = nn.Linear(128, 3)


    def forward(self, encoded, viewdirs, density_noise_std=0.0, generator=None,
                regnerf_density=False):

        x = encoded

        for i, layer in enumerate(self.layers):
            if i == 4:
                x = torch.cat([x, encoded], dim=-1)

            x = F.relu(layer(x))

        raw_density = self.density_head(x)
        if self.training and density_noise_std:
            noise_shape = () if regnerf_density else raw_density.shape
            noise = torch.randn(noise_shape, device=x.device, dtype=raw_density.dtype,
                                generator=generator)
            raw_density = raw_density + density_noise_std * noise
        density = F.softplus(raw_density - 1.0 if regnerf_density else raw_density)
        if viewdirs is None:
            return None, density

        features = self.feature_head(x)

        viewdirs = F.normalize(viewdirs, dim=-1)

        dir_encoded = positional_encoding(viewdirs)
        dir_encoded = dir_encoded[:, None, :].expand(
            -1, encoded.shape[1], -1
        )

        color_input = torch.cat([features, dir_encoded], dim=-1)

        color_hidden = F.relu(self.color_layer(color_input))

        rgb = torch.sigmoid(self.color_head(color_hidden))

        return rgb, density



def ray_depth(weights, intervals):
    midpoints = 0.5 * (intervals[..., :-1] + intervals[..., 1:])
    opacity = weights.sum(dim=-1)
    depth = (weights * midpoints).sum(dim=-1) / opacity.clamp_min(torch.finfo(weights.dtype).eps)
    return depth.clamp(min=intervals[..., 0], max=intervals[..., -1]), opacity


class MipNeRF(nn.Module):
    def __init__(self, num_samples=64, near=2.0, far=6.0, background="white", density_noise_std=0.0,
                 regnerf_density=False):
        super().__init__()

        self.num_samples = num_samples
        self.near = near
        self.far = far
        self.set_extra_state({"background": background, "density_noise_std": density_noise_std,
                              "regnerf_density": regnerf_density})

        # Shared by coarse and fine passes
        self.mlp = MipNeRFMLP()

    def get_extra_state(self):
        return {"background": self.background, "density_noise_std": self.density_noise_std,
                "regnerf_density": self.regnerf_density}

    def set_extra_state(self, state):
        background = state["background"]
        noise = state["density_noise_std"]
        if background not in ("white", "black"):
            raise ValueError("Use white or black for the background.")
        if not math.isfinite(noise) or noise < 0:
            raise ValueError("Density noise must be finite and nonnegative.")
        self.background = background
        self.density_noise_std = noise
        self.regnerf_density = state.get("regnerf_density", False)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # Earlier checkpoints always used white and had no density noise.
        state_dict.setdefault(prefix + "_extra_state", {"background": "white", "density_noise_std": 0.0})
        super()._load_from_state_dict(state_dict, prefix, *args)

    def render_pass(self, rays_o, rays_d, radii, t_vals, generator=None, depth_only=False):
        # Convert cone sections into Gaussians
        means, covs = conical_frustum_to_gaussian(
            rays_o, rays_d, t_vals, radii
        )

        # Integrated positional encoding
        encoded = integrated_positional_encoding(
            means, covs
        )

        # Predict RGB and density
        viewdirs = None if depth_only else rays_d
        rgb, density = self.mlp(encoded, viewdirs, self.density_noise_std, generator,
                                self.regnerf_density)

        # Combine samples into pixel colors
        rgb_map, weights = volume_render(
            rgb, density, t_vals, rays_d, white_bkgd=self.background == "white"
        )

        return rgb_map, weights

    def forward(self, rays_o, rays_d, radii, return_depth=False, *, return_depth_and_opacity=False,
                return_depth_levels=False, return_patch_rgb=False, generator=None):
        # Expect a batch of flattened rays
        # rays_o: [B, 3]
        # rays_d: [B, 3]
        # radii:  [B, 1]

        B = rays_o.shape[0]

        # 1. Coarse sampling
        t_coarse = sample_intervals(
            B,
            near=self.near,
            far=self.far,
            N=self.num_samples,
            device=rays_o.device,
            dtype=rays_o.dtype
        )

        # 2. Coarse rendering
        coarse_rgb, coarse_weights = self.render_pass(
            rays_o, rays_d, radii, t_coarse, generator, depth_only=return_depth_levels
        )

        # 3. Fine sampling
        t_fine = resample_intervals(
            t_coarse,
            coarse_weights,
            N=self.num_samples,
            randomized=self.training, generator=generator
        )

        # 4. Fine rendering (same MLP)
        fine_rgb, fine_weights = self.render_pass(
            rays_o, rays_d, radii, t_fine, generator, depth_only=return_depth_levels and not return_patch_rgb
        )

        if return_depth_levels:
            levels = ray_depth(coarse_weights, t_coarse), ray_depth(fine_weights, t_fine)
            return (levels, fine_rgb.detach()) if return_patch_rgb else levels
        if return_depth or return_depth_and_opacity:
            fine_depth, opacity = ray_depth(fine_weights, t_fine)
            if return_depth_and_opacity:
                return coarse_rgb, fine_rgb, fine_depth, opacity
            return coarse_rgb, fine_rgb, fine_depth

        return coarse_rgb, fine_rgb
