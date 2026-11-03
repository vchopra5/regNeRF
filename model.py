"""
NeRF        -> march along a ray, sample points, encode each point with sin/cos,
               run MLP to get colour and density, and alpha-composite
Mip-NeRF    -> instead of using a ray (line), use a frustum (cone)
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


# sample_intervals gives only the coarse slices, which are evenly spaced and
# know nothing about the scene. The fine slices come from resample_intervals,
# which moves them toward where the coarse pass found a surface.
def sample_intervals(
    num_rays,
    near=2.0,
    far=6.0,
    N=64,
    device=None,
    dtype=torch.float32,
    randomized=False,
    generator=None,
):
    # NeRF sampled 64 points per ray
    # Mip-NeRF samples 64 intervals [t0, t1] which are slices of the cone
    t_vals = torch.linspace(near, far, N + 1, device=device, dtype=dtype)
    t_vals = t_vals.expand(num_rays, -1)
    if randomized:
        # Mip-NeRF stratified sampling: move each boundary to a random point
        # between its neighbouring midpoints. The first and last boundaries
        # move inward by up to half a step, so all samples stay in [near, far].
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mids], dim=-1)
        u = torch.rand(t_vals.shape, device=device, dtype=dtype, generator=generator)
        t_vals = lower + (upper - lower) * u
    return t_vals


# The pixel's cone has radius radii * t at distance t. One interval [t₀, t₁]
# cuts out a conical frustum, a cone with its tip cut off.
#
# The problem: a frustum is a uniform solid with flat ends and a slanted side.
# The average of sin over that shape has no simple formula. You would need numerical
# integration for every sample, every frequency, and every step.
#
# We want a Gaussian with the same mean and the same spread as that frustum.
def conical_frustum_to_gaussian(rays_o, rays_d, t_vals, radii):
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    radii = radii.reshape(-1, 1)

    t0, t1 = t_vals[..., :-1], t_vals[..., 1:]  # [B,N] each
    t_mu = (t0 + t1) / 2  # Interval centre
    t_delta = (t1 - t0) / 2  # Interval half-width

    # The mean sits slightly past the centre, because the far
    # end of the frustum is wider and holds more volume.
    denom = 3 * (t_mu**2) + (t_delta**2)
    mean_t = t_mu + (2 * t_mu * t_delta**2) / denom

    # Spread along the ray, about the interval length
    var_t = (
        t_delta**2 / 3
        - (4 / 15) * (t_delta**4 * (12 * t_mu**2 - t_delta**2)) / denom**2
    )

    # Spread across the ray. It grows with t, because the cone widens.
    var_r = radii**2 * (
        t_mu**2 / 4 + 5 * t_delta**2 / 12 - (4 / 15) * t_delta**4 / denom
    )

    # Convert to 3D world space
    means = rays_o[:, None, :] + rays_d[:, None, :] * mean_t[..., None]

    d_sq = rays_d**2
    d_mag_sq = d_sq.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    cov_diag = (
        var_t[..., None] * d_sq[:, None, :]
        + var_r[..., None] * (1 - d_sq / d_mag_sq)[:, None, :]
    )

    return means, cov_diag


# For the conical rays
def integrated_positional_encoding(means, cov_diag, min_deg=0, max_deg=16):
    # NeRF maps a point to sin(2 ** l * x), cos(2 ** l * x) but this causes aliasing
    # Instead we encode the average of each feature over the Gaussian in closed-form

    scales = 2.0 ** torch.arange(
        min_deg, max_deg, device=means.device, dtype=means.dtype
    )

    scaled_means = means[..., None, :] * scales[None, None, :, None]
    scaled_vars = cov_diag[..., None, :] * scales[None, None, :, None] ** 2

    # Separate weight for each frequency, and it compares the
    # Gaussian's size with that frequency's wavelength (like a low-pass filter)
    #
    # Near, small Gaussian -> ≈ 1 -> The feature is the same as NeRF's
    # Far, large Gaussian -> ≈ 0 -> The feature is removed
    attenuation = torch.exp(-0.5 * scaled_vars)

    sin_features = torch.sin(scaled_means) * attenuation
    cos_features = torch.cos(scaled_means) * attenuation

    sin_features = sin_features.flatten(start_dim=-2)
    cos_features = cos_features.flatten(start_dim=-2)

    return torch.cat([sin_features, cos_features], dim=-1)


# For viewing directions
# IPE says where the sample is and how big it is; PE says which way the camera is looking
# This is because a direction is one exact value with no volume, so there is nothing to average
def positional_encoding(x, num_freqs=4):
    scales = 2 ** (torch.arange(num_freqs, device=x.device, dtype=x.dtype))

    scaled = x[..., None, :] * scales[:, None]

    return torch.cat(
        [x, torch.sin(scaled).flatten(-2), torch.cos(scaled).flatten(-2)], dim=-1
    )


# NeRF's rendering equation with no conceptual change
def volume_render(rgb, density, t_vals, rays_d, white_bkgd=True):

    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = delta * torch.linalg.norm(rays_d, dim=-1, keepdim=True)

    sigma = density.squeeze(-1)
    alpha = 1.0 - torch.exp(-sigma * delta)

    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]

    weights = alpha * transmittance

    if rgb is None:
        return None, weights

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    acc = weights.sum(dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1 - acc[..., None])

    return rgb_map, weights


# NeRF's two-pass idea is unchanged: the coarse pass finds where the surface is,
# and the fine pass looks there closely. What changes is how the fine samples are placed
# Coarse -> fine bridge: place the fine slices where the coarse weights are high.
def resample_intervals(t_vals, weights, N=64, randomized=True, generator=None):
    """
    t_vals:  [B, M+1] coarse interval boundaries
    weights: [B, M]   coarse rendering weights

    Returns: [B, N+1] new interval boundaries
    """

    # Smooth the weights so nearby regions are considered
    weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)

    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])

    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # Avoid zero-probability intervals
    weights_blur = weights_blur + 0.01

    # Normalize weights into probabilities
    pdf = weights_blur / weights_blur.sum(dim=-1, keepdim=True)

    # Build cumulative distribution
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # Sample N+1 quantiles between 0 and 1
    if randomized:
        u = (
            torch.arange(N + 1, device=t_vals.device, dtype=t_vals.dtype)[None, :]
            + torch.rand(
                t_vals.shape[0],
                N + 1,
                device=t_vals.device,
                generator=generator,
                dtype=t_vals.dtype,
            )
        ) / (N + 1)

        u[..., 0] = 0.0
        u[..., -1] = 1.0
    else:
        u = torch.linspace(
            0.0, 1.0, N + 1, device=t_vals.device, dtype=t_vals.dtype
        ).expand(t_vals.shape[0], -1)

    # Find which CDF interval contains each sample
    inds = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)

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
        # 8 layers × 256 with ReLU. The input is concatenated in again at layer index 4
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(depth):
            in_dim = pos_dim if i == 0 else hidden_dim

            if i == 4:
                in_dim += pos_dim

            self.layers.append(nn.Linear(in_dim, hidden_dim))

        self.density_head = nn.Linear(hidden_dim, 1)
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        dir_dim = 3 + 4 * 3 * 2

        self.color_layer = nn.Linear(hidden_dim + dir_dim, 128)
        self.color_head = nn.Linear(128, 3)

    def forward(
        self,
        encoded,
        viewdirs,
        density_noise_std=0.0,
        generator=None,
        regnerf_density=False,
    ):

        x = encoded

        for i, layer in enumerate(self.layers):
            if i == 4:
                x = torch.cat([x, encoded], dim=-1)

            x = F.relu(layer(x))

        raw_density = self.density_head(x)
        # Noise and the -1 shift both discourage fog that can fit only a few views.
        if self.training and density_noise_std:
            # regnerf_density: one noise value for the whole batch (NeRF: one per sample). Check vs RegNeRF.
            noise_shape = () if regnerf_density else raw_density.shape
            noise = torch.randn(
                noise_shape,
                device=x.device,
                dtype=raw_density.dtype,
                generator=generator,
            )
            raw_density = raw_density + density_noise_std * noise
        # Softplus instead of NeRF's ReLU for density head; the -1 shift starts the scene closer to empty.
        density = F.softplus(raw_density - 1.0 if regnerf_density else raw_density)
        # Depth-only mode for RegNeRF patches: skip the colour branch.
        if viewdirs is None:
            return None, density

        features = self.feature_head(x)

        viewdirs = F.normalize(viewdirs, dim=-1)

        dir_encoded = positional_encoding(viewdirs)
        dir_encoded = dir_encoded[:, None, :].expand(-1, encoded.shape[1], -1)

        color_input = torch.cat([features, dir_encoded], dim=-1)

        color_hidden = F.relu(self.color_layer(color_input))

        rgb = torch.sigmoid(self.color_head(color_hidden))

        return rgb, density


# Like NeRF's depth map, but normalized by opacity and returned with it for RegNeRF's loss.
def ray_depth(weights, intervals):
    midpoints = 0.5 * (intervals[..., :-1] + intervals[..., 1:])
    opacity = weights.sum(dim=-1)
    depth = (weights * midpoints).sum(dim=-1) / opacity.clamp_min(
        torch.finfo(weights.dtype).eps
    )
    return depth.clamp(min=intervals[..., 0], max=intervals[..., -1]), opacity


class MipNeRF(nn.Module):
    def __init__(
        self,
        num_samples=64,
        near=2.0,
        far=6.0,
        background="white",
        density_noise_std=0.0,
        regnerf_density=False,
        stratified=True,
    ):
        super().__init__()

        # near/far are not saved: train.py anneals them, so loaders must pass them.
        self.num_samples = num_samples
        self.near = near
        self.far = far
        self.set_extra_state(
            {
                "background": background,
                "density_noise_std": density_noise_std,
                "regnerf_density": regnerf_density,
                "stratified": stratified,
            }
        )

        # Shared by coarse and fine passes
        self.mlp = MipNeRFMLP()

    # Saved with the weights, so a loaded model renders the way it trained.
    def get_extra_state(self):
        return {
            "background": self.background,
            "density_noise_std": self.density_noise_std,
            "regnerf_density": self.regnerf_density,
            "stratified": self.stratified,
        }

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
        # Checkpoints saved before this flag existed trained without coarse jitter.
        self.stratified = state.get("stratified", False)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # Earlier checkpoints always used white and had no density noise.
        state_dict.setdefault(
            prefix + "_extra_state", {"background": "white", "density_noise_std": 0.0}
        )
        super()._load_from_state_dict(state_dict, prefix, *args)

    # One level: frustum -> Gaussian -> IPE -> MLP -> volume render.
    def render_pass(
        self, rays_o, rays_d, radii, t_vals, generator=None, depth_only=False
    ):
        # Convert cone sections into Gaussians
        means, covs = conical_frustum_to_gaussian(rays_o, rays_d, t_vals, radii)

        # Integrated positional encoding
        encoded = integrated_positional_encoding(means, covs)

        # Predict RGB and density
        viewdirs = None if depth_only else rays_d
        rgb, density = self.mlp(
            encoded, viewdirs, self.density_noise_std, generator, self.regnerf_density
        )

        # Combine samples into pixel colors
        rgb_map, weights = volume_render(
            rgb, density, t_vals, rays_d, white_bkgd=self.background == "white"
        )

        return rgb_map, weights

    def forward(
        self,
        rays_o,
        rays_d,
        radii,
        return_depth=False,
        *,
        return_depth_levels=False,
        generator=None,
    ):
        # Expect a batch of flattened rays
        # rays_o: [B, 3]
        # rays_d: [B, 3]
        #   Not normalized, in the form (x/f, y/f, -1) so t measures depth along camera axis
        # radii:  [B, 1]
        #   Cone grows linearly with distance, so its radius at distance t is radii * t

        # generator: train.py passes depth_rng for depth patches only.
        B = rays_o.shape[0]

        # 1. Coarse sampling (jitter only if training and stratified)
        t_coarse = sample_intervals(
            B,
            near=self.near,
            far=self.far,
            N=self.num_samples,
            device=rays_o.device,
            dtype=rays_o.dtype,
            randomized=self.training and self.stratified,
            generator=generator,
        )

        # 2. Coarse rendering
        coarse_rgb, coarse_weights = self.render_pass(
            rays_o, rays_d, radii, t_coarse, generator, depth_only=return_depth_levels
        )

        # 3. Fine sampling (jitter whenever training)
        t_fine = resample_intervals(
            t_coarse,
            coarse_weights,
            N=self.num_samples,
            randomized=self.training,
            generator=generator,
        )

        # 4. Fine rendering (same MLP)
        fine_rgb, fine_weights = self.render_pass(
            rays_o, rays_d, radii, t_fine, generator, depth_only=return_depth_levels
        )

        # Depth levels: RegNeRF depth patches, no colour. return_depth: evaluation.
        if return_depth_levels:
            return ray_depth(coarse_weights, t_coarse), ray_depth(fine_weights, t_fine)
        if return_depth:
            return coarse_rgb, fine_rgb, ray_depth(fine_weights, t_fine)[0]

        return coarse_rgb, fine_rgb


def render_rays(model, rays, chunk_size, device=None, *, return_depth=False):
    """Render flattened rays in chunks. Return fine RGB, plus fine depth if requested."""
    outputs = []
    for batch in zip(*(ray.split(chunk_size) for ray in rays)):
        outputs.append(
            model(*(ray.to(device) for ray in batch), return_depth=return_depth)[1:]
        )
    parts = [torch.cat(items) for items in zip(*outputs)]
    return tuple(parts) if return_depth else parts[0]


def load_model(path, device, **settings):
    """Load a final state dict or a training checkpoint for evaluation."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = MipNeRF(**settings)
    model.load_state_dict(saved.get("model", saved))
    return model.to(device).eval()
