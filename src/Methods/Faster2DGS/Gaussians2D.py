"""Faster2DGS/Gaussians2D.py: 2D surfel Gaussian primitives (N×2 log-scales)."""

from __future__ import annotations

import math

import numpy as np
import torch

import Framework
from Cameras.utils import quaternion_to_rotation_matrix
from Datasets.utils import BasicPointCloud
from Logging import Logger
from Methods.FasterGS.FasterGSCudaBackend import add_noise, relocation_adjustment
from Methods.FasterGS.Model import Gaussians
from Optim.adam_utils import extend_param_groups, replace_param_group_data
from Optim.knn_utils import compute_root_mean_squared_knn_distances

# Fixed log-scale for the implicit thin axis (matches Faster2DGSRenderer / surfel bridge).
DEFAULT_Z_LOG_SCALE_COMPAT = -6.0


class Gaussians2D(Gaussians):
    """Oriented 2D disks: ``(N, 2)`` log-scales + quaternion tangent frame."""

    z_log_scale_compat: float = DEFAULT_Z_LOG_SCALE_COMPAT
    # 3DGS/2DGS default is 0.01; large surfel clouds stay too dark until opacities recover.
    opacity_reset_max: float = 0.05

    @property
    def scales(self) -> torch.Tensor:
        return self._scales.exp()

    @property
    def raw_scales(self) -> torch.Tensor:
        return self._scales

    @property
    def raw_scales_2d(self) -> torch.Tensor:
        return self._scales

    @property
    def raw_scales_3d_compat(self) -> torch.Tensor:
        """Expand to ``(N, 3)`` log-scales for legacy 3D CUDA paths (EWA fallback, MCMC noise)."""
        z = torch.full(
            (self._scales.shape[0], 1),
            self.z_log_scale_compat,
            device=self._scales.device,
            dtype=self._scales.dtype,
        )
        return torch.cat([self._scales, z], dim=1)

    @property
    def covariances(self) -> torch.Tensor:
        """Surfel-style covariance: ``R @ diag(s_x, s_y, 1)`` embedded in 3D."""
        R = quaternion_to_rotation_matrix(self.rotations, normalize=False)
        s = self.scales
        S = torch.zeros((s.shape[0], 3, 3), device=s.device, dtype=s.dtype)
        S[:, 0, 0] = s[:, 0]
        S[:, 1, 1] = s[:, 1]
        S[:, 2, 2] = 1.0
        RS = R @ S
        return RS @ RS.transpose(-2, -1)

    def setup_3d_filter(self, filter_config: Framework.ConfigParameterList, dataset) -> None:
        Logger.log_warning('Gaussians2D ignores FILTER_3D (not applicable to 2D surfel scales).')

    def initialize_from_point_cloud(self, point_cloud: BasicPointCloud, use_mcmc: bool) -> None:
        means = point_cloud.positions.cuda()
        n_initial_gaussians = means.shape[0]
        Logger.log_info(f'number of 2D Gaussians at initialization: {n_initial_gaussians:,}')
        rgbs = torch.full_like(means, fill_value=0.5) if point_cloud.colors is None else point_cloud.colors.cuda()
        sh_coefficients_0 = ((rgbs - 0.5) / 0.28209479177387814)[:, None, :]
        sh_coefficients_rest = torch.zeros(
            (n_initial_gaussians, (self.max_sh_degree + 1) ** 2 - 1, 3),
            dtype=torch.float32,
            device='cuda',
        )
        distances = compute_root_mean_squared_knn_distances(means)
        distances = distances * 0.1 if use_mcmc else distances
        scales = distances.log()[..., None].repeat(1, 2)
        # 2DGS: random unit quaternions, not identity (identity aligns all disk normals).
        rotations = torch.rand((n_initial_gaussians, 4), dtype=torch.float32, device='cuda')
        initial_opacity = 0.5 if use_mcmc else 0.1
        initial_opacity_logit = math.log(initial_opacity / (1.0 - initial_opacity))
        opacities = torch.full((n_initial_gaussians, 1), fill_value=initial_opacity_logit, dtype=torch.float32, device='cuda')
        self._means = torch.nn.Parameter(means.contiguous())
        self._sh_coefficients_0 = torch.nn.Parameter(sh_coefficients_0.contiguous())
        self._sh_coefficients_rest = torch.nn.Parameter(sh_coefficients_rest.contiguous())
        self._scales = torch.nn.Parameter(scales.contiguous())
        self._rotations = torch.nn.Parameter(rotations.contiguous())
        self._opacities = torch.nn.Parameter(opacities.contiguous())
        self._max_radii2D = torch.zeros(n_initial_gaussians, dtype=torch.float32, device='cuda')

    @torch.no_grad()
    def reset_opacities(self) -> None:
        """Cap activated opacity at ``opacity_reset_max`` (official 2DGS uses 0.01)."""
        max_opacity = min(max(float(self.opacity_reset_max), 1e-4), 1.0 - 1e-4)
        max_logit = math.log(max_opacity / (1.0 - max_opacity))
        opacities_new = self._opacities.clamp_max(max_logit)
        replace_param_group_data(self.optimizer, opacities_new, 'opacities')

    @torch.no_grad()
    def update_max_radii2D(self, radii: torch.Tensor) -> None:
        """Track per-Gaussian max screen radius (official 2DGS densify_and_prune)."""
        if self._max_radii2D is None or radii.numel() == 0:
            return
        visible = radii > 0
        if not bool(visible.any().item()):
            return
        self._max_radii2D[visible] = torch.max(self._max_radii2D[visible], radii[visible].float())

    def prune(self, prune_mask: torch.Tensor) -> None:
        if self._max_radii2D is not None:
            self._max_radii2D = self._max_radii2D[~prune_mask].contiguous()
        super().prune(prune_mask)

    def sort(self, ordering: torch.Tensor) -> None:
        if self._max_radii2D is not None:
            self._max_radii2D = self._max_radii2D[ordering].contiguous()
        super().sort(ordering)

    def adaptive_density_control(
        self,
        grad_threshold: float,
        min_opacity: float,
        prune_large_gaussians: bool,
        max_screen_size: float | None = None,
        *,
        skip_opacity_prune: bool = False,
    ) -> None:
        """Clone/split with 2DGS-style in-plane sampling (zero extent along local normal)."""
        # reset_opacities() clamps to ~0.01; culling above that evicts the whole cloud right after 3k.
        if not skip_opacity_prune:
            min_opacity = min(float(min_opacity), 0.01)
        n_before = self._means.shape[0]
        densification_mask = self.densification_info[1] >= grad_threshold * self.densification_info[0].clamp_min(1.0)
        is_small = torch.max(self._scales, dim=1).values <= math.log(self.percent_dense * self.training_cameras_extent)

        duplicate_mask = densification_mask & is_small
        n_new_gaussians_duplicate = duplicate_mask.sum().item()
        duplicated_means = self._means[duplicate_mask]
        duplicated_sh_coefficients_0 = self._sh_coefficients_0[duplicate_mask]
        duplicated_sh_coefficients_rest = self._sh_coefficients_rest[duplicate_mask]
        duplicated_opacities = self._opacities[duplicate_mask]
        duplicated_scales = self._scales[duplicate_mask]
        duplicated_rotations = self._rotations[duplicate_mask]

        split_mask = densification_mask & ~is_small
        n_new_gaussians_split = 2 * split_mask.sum().item()
        if split_mask.any():
            split_scales_2d = self._scales[split_mask].exp().repeat(2, 1)
            split_rotations = self._rotations[split_mask].repeat(2, 1)
            samples_xy = torch.randn_like(split_scales_2d) * split_scales_2d
            samples = torch.cat([samples_xy, torch.zeros(samples_xy.shape[0], 1, device=samples_xy.device)], dim=1)
            offsets = (quaternion_to_rotation_matrix(split_rotations) @ samples[..., None])[..., 0]
            split_means = self._means[split_mask].repeat(2, 1) + offsets
            split_scales = split_scales_2d.div(1.6).log()  # official 2DGS: / (0.8 * N), N=2
            split_sh_coefficients_0 = self._sh_coefficients_0[split_mask].repeat(2, 1, 1)
            split_sh_coefficients_rest = self._sh_coefficients_rest[split_mask].repeat(2, 1, 1)
            split_opacities = self._opacities[split_mask].repeat(2, 1)
        else:
            split_means = self._means.new_empty((0, 3))
            split_scales = self._scales.new_empty((0, 2))
            split_sh_coefficients_0 = self._sh_coefficients_0.new_empty((0, *self._sh_coefficients_0.shape[1:]))
            split_sh_coefficients_rest = self._sh_coefficients_rest.new_empty((0, *self._sh_coefficients_rest.shape[1:]))
            split_opacities = self._opacities.new_empty((0, 1))
            split_rotations = self._rotations.new_empty((0, 4))

        saved_max_radii2D = self._max_radii2D.clone() if self._max_radii2D is not None else None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        param_groups = extend_param_groups(self.optimizer, {
            'means': torch.cat([duplicated_means, split_means], dim=0),
            'sh_coefficients_0': torch.cat([duplicated_sh_coefficients_0, split_sh_coefficients_0], dim=0),
            'sh_coefficients_rest': torch.cat([duplicated_sh_coefficients_rest, split_sh_coefficients_rest], dim=0),
            'opacities': torch.cat([duplicated_opacities, split_opacities], dim=0),
            'scales': torch.cat([duplicated_scales, split_scales], dim=0),
            'rotations': torch.cat([duplicated_rotations, split_rotations], dim=0),
        })
        del duplicated_means, duplicated_sh_coefficients_0, duplicated_sh_coefficients_rest
        del duplicated_opacities, duplicated_scales, duplicated_rotations
        del split_means, split_scales, split_sh_coefficients_0, split_sh_coefficients_rest
        del split_opacities, split_rotations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._means = param_groups['means']
        self._sh_coefficients_0 = param_groups['sh_coefficients_0']
        self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
        self._opacities = param_groups['opacities']
        self._scales = param_groups['scales']
        self._rotations = param_groups['rotations']

        n_new_gaussians = n_new_gaussians_duplicate + n_new_gaussians_split
        if self._max_radii2D is not None:
            self._max_radii2D = torch.zeros(self._means.shape[0], dtype=torch.float32, device='cuda')

        self._densification_info = None
        self._filter_3d = None

        prune_mask = torch.cat([split_mask, torch.zeros(n_new_gaussians, dtype=torch.bool, device='cuda')])
        opacity_prune = torch.zeros_like(prune_mask)
        if not skip_opacity_prune:
            opacity_thresh = math.log(min_opacity / (1 - min_opacity))
            opacity_prune = self._opacities.flatten() < opacity_thresh
            prune_mask |= opacity_prune
        rot_prune = self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        prune_mask |= rot_prune
        prune_mask |= rot_prune
        world_prune = torch.zeros_like(prune_mask)
        screen_prune = torch.zeros_like(prune_mask)
        if prune_large_gaussians:
            world_thresh = math.log(0.1 * self.training_cameras_extent)
            world_prune = self._scales.max(dim=1).values > world_thresh
            prune_mask |= world_prune
            if max_screen_size is not None and saved_max_radii2D is not None:
                n_old = saved_max_radii2D.shape[0]
                if self._max_radii2D.shape[0] == n_old + n_new_gaussians:
                    screen_radii = torch.cat([
                        saved_max_radii2D,
                        torch.zeros(n_new_gaussians, device='cuda', dtype=saved_max_radii2D.dtype),
                    ])
                    screen_prune = screen_radii > max_screen_size
                    prune_mask |= screen_prune
        n_pruned = int(prune_mask.sum().item())
        self._last_prune_breakdown = {
            'split_parents': int(split_mask.sum().item()),
            'opacity': int(opacity_prune.sum().item()),
            'rotation': int(rot_prune.sum().item()),
            'world_scale': int(world_prune.sum().item()),
            'screen_size': int(screen_prune.sum().item()),
            'total': n_pruned,
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.prune(prune_mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._last_densify_stats = {
            'n_before': n_before,
            'n_duplicate': n_new_gaussians_duplicate,
            'n_split_parents': int(split_mask.sum().item()),
            'n_new': n_new_gaussians,
            'n_pruned': n_pruned,
            'n_after': int(self._means.shape[0]),
            'grad_candidates': int(densification_mask.sum().item()),
        }

    def mcmc_densification(self, min_opacity: float, cap_max: int) -> None:
        """MCMC relocation/add using 3D-compat scales internally, storing ``(N, 2)`` log-scales."""
        dead_mask = self._opacities.flatten() <= math.log(min_opacity / (1 - min_opacity))
        dead_mask |= self._rotations.mul(self._rotations).sum(dim=1) < 1e-8
        n_dead_gaussians = dead_mask.sum().item()
        if n_dead_gaussians > 0:
            dead_indices = torch.where(dead_mask)[0]
            alive_indices = torch.where(~dead_mask)[0]
            opacities = self.opacities.flatten()
            sampled_indices = torch.multinomial(opacities[alive_indices], n_dead_gaussians, replacement=True)
            sampled_indices = alive_indices[sampled_indices]
            _, inverse, counts_per_unique = sampled_indices.unique(sorted=False, return_inverse=True, return_counts=True)
            counts = counts_per_unique[inverse] + 1
            adjusted_opacities, adjusted_scales_3d = relocation_adjustment(
                opacities[sampled_indices],
                torch.cat([self.scales[sampled_indices], torch.ones(sampled_indices.shape[0], 1, device='cuda')], dim=1),
                counts,
            )
            adjusted_opacities = adjusted_opacities.clamp(min_opacity, 1.0 - torch.finfo(torch.float32).eps).logit()
            adjusted_scales = adjusted_scales_3d[:, :2].log()
            self._opacities[sampled_indices] = adjusted_opacities
            self._scales[sampled_indices] = adjusted_scales
            self._means[dead_indices] = self._means[sampled_indices]
            self._sh_coefficients_0[dead_indices] = self._sh_coefficients_0[sampled_indices]
            self._sh_coefficients_rest[dead_indices] = self._sh_coefficients_rest[sampled_indices]
            self._opacities[dead_indices] = adjusted_opacities
            self._scales[dead_indices] = adjusted_scales
            self._rotations[dead_indices] = self._rotations[sampled_indices]
            from Optim.adam_utils import reset_state
            reset_state(self.optimizer, indices=sampled_indices)
            self._densification_info = None
            self._filter_3d = None

        current_n_points = self._means.shape[0]
        n_target = min(cap_max, int(1.05 * current_n_points))
        n_added_gaussians = max(0, n_target - current_n_points)
        if n_added_gaussians > 0:
            opacities = self.opacities.flatten()
            sampled_indices = torch.multinomial(opacities, n_added_gaussians, replacement=True)
            _, inverse, counts_per_unique = sampled_indices.unique(sorted=False, return_inverse=True, return_counts=True)
            counts = counts_per_unique[inverse] + 1
            adjusted_opacities, adjusted_scales_3d = relocation_adjustment(
                opacities[sampled_indices],
                torch.cat([self.scales[sampled_indices], torch.ones(sampled_indices.shape[0], 1, device='cuda')], dim=1),
                counts,
            )
            adjusted_opacities = adjusted_opacities.clamp(min_opacity, 1.0 - torch.finfo(torch.float32).eps).logit()
            adjusted_scales = adjusted_scales_3d[:, :2].log()
            self._opacities[sampled_indices] = adjusted_opacities
            self._scales[sampled_indices] = adjusted_scales
            param_groups = extend_param_groups(self.optimizer, {
                'means': self._means[sampled_indices],
                'sh_coefficients_0': self._sh_coefficients_0[sampled_indices],
                'sh_coefficients_rest': self._sh_coefficients_rest[sampled_indices],
                'opacities': adjusted_opacities,
                'scales': adjusted_scales,
                'rotations': self._rotations[sampled_indices],
            })
            self._means = param_groups['means']
            self._sh_coefficients_0 = param_groups['sh_coefficients_0']
            self._sh_coefficients_rest = param_groups['sh_coefficients_rest']
            self._opacities = param_groups['opacities']
            self._scales = param_groups['scales']
            self._rotations = param_groups['rotations']
            from Optim.adam_utils import reset_state
            reset_state(self.optimizer, indices=sampled_indices)
            self._densification_info = None
            self._filter_3d = None

    @torch.no_grad()
    def post_optimizer_step(self, inject_noise: bool) -> None:
        if inject_noise:
            add_noise(
                self.raw_scales_3d_compat,
                self.raw_rotations,
                self.raw_opacities,
                self.means,
                5e5 * self.lr_means,
            )

    @torch.no_grad()
    def as_ply_dict(self) -> dict[str, np.ndarray]:
        if self.means.shape[0] == 0:
            return {}
        means = self.means.detach().contiguous().cpu().numpy()
        sh_0 = self.sh_coefficients_0.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sh_rest = self.sh_coefficients_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.raw_opacities.detach().contiguous().cpu().numpy()
        scales = self.raw_scales.detach().contiguous().cpu().numpy()
        rotations = self.rotations.detach().contiguous().cpu().numpy()
        attributes = np.concatenate((means, sh_0, sh_rest, opacities, scales, rotations), axis=1)
        attribute_names = (
            ['x', 'y', 'z']
            + ['f_dc_0', 'f_dc_1', 'f_dc_2']
            + [f'f_rest_{i}' for i in range(sh_rest.shape[-1])]
            + ['opacity']
            + ['scale_0', 'scale_1']
            + ['rot_0', 'rot_1', 'rot_2', 'rot_3']
        )
        dtype = 'f4'
        full_dtype = [(name, dtype) for name in attribute_names]
        vertices = np.empty(means.shape[0], dtype=full_dtype)
        vertices[:] = list(map(tuple, attributes))
        return {'vertex': vertices}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        scales_key = prefix + '_scales'
        if scales_key in state_dict and state_dict[scales_key].shape[-1] == 3:
            Logger.log_info('Loading legacy 3D checkpoint into Gaussians2D: keeping XY log-scales only.')
            state_dict[scales_key] = state_dict[scales_key][..., :2].clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
