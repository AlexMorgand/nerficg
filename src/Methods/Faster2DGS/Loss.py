"""Faster2DGS/Loss.py"""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Methods.Faster2DGS.Model import Faster2DGSModel
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DepthSmoothness import depth_smoothness_loss
from Optim.Losses.DSSIM import fused_dssim


class Faster2DGSLoss(BaseLoss):
    """Loss container with 2DGS-style geometric regularization hooks."""

    def __init__(self, loss_config: ConfigParameterList, model: Faster2DGSModel) -> None:
        super().__init__()
        self._gaussians = model.gaussians
        self._render_pkg: dict[str, torch.Tensor] | None = None
        self._rgb_guide: torch.Tensor | None = None
        self._base_distortion_weight = float(loss_config.LAMBDA_DISTORTION)
        self._base_normal_weight = float(loss_config.LAMBDA_NORMAL)
        self._base_depth_smoothness_weight = float(getattr(loss_config, 'LAMBDA_DEPTH_SMOOTHNESS', 0.0))
        self._base_planar_splat_weight = float(loss_config.LAMBDA_PLANAR_SPLAT)
        self._distortion_alpha_mask_min = float(getattr(loss_config, 'DISTORTION_ALPHA_MASK_MIN', 0.05))
        self._last_geometry_log: dict[str, float] = {}
        self.add_loss_metric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)
        self.add_loss_metric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.add_loss_metric('DISTORTION_REGULARIZATION', self.distortion_regularization_loss, loss_config.LAMBDA_DISTORTION)
        self.add_loss_metric('NORMAL_REGULARIZATION', self.normal_regularization_loss, loss_config.LAMBDA_NORMAL)
        self.add_loss_metric(
            'DEPTH_SMOOTHNESS_REGULARIZATION',
            self.depth_smoothness_regularization_loss,
            getattr(loss_config, 'LAMBDA_DEPTH_SMOOTHNESS', 0.0),
        )
        self.add_loss_metric('PLANAR_SPLAT_REGULARIZATION', self.planar_splat_regularization_loss, loss_config.LAMBDA_PLANAR_SPLAT)
        self.add_loss_metric('OPACITY_REGULARIZATION', model.gaussians.opacity_regularization_loss, loss_config.LAMBDA_OPACITY_REGULARIZATION)
        self.add_loss_metric('SCALE_REGULARIZATION', model.gaussians.scale_regularization_loss, loss_config.LAMBDA_SCALE_REGULARIZATION)
        self.add_quality_metric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def set_geometry_loss_weights(
        self,
        *,
        distortion_scale: float = 1.0,
        normal_scale: float = 1.0,
        depth_smoothness_scale: float = 1.0,
        planar_splat_scale: float = 1.0,
    ) -> None:
        for metric in self.loss_metrics:
            if metric.name == 'DISTORTION_REGULARIZATION':
                metric.weight = max(0.0, self._base_distortion_weight * float(distortion_scale))
            elif metric.name == 'NORMAL_REGULARIZATION':
                metric.weight = max(0.0, self._base_normal_weight * float(normal_scale))
            elif metric.name == 'DEPTH_SMOOTHNESS_REGULARIZATION':
                metric.weight = max(0.0, self._base_depth_smoothness_weight * float(depth_smoothness_scale))
            elif metric.name == 'PLANAR_SPLAT_REGULARIZATION':
                metric.weight = max(0.0, self._base_planar_splat_weight * float(planar_splat_scale))

    def _zero(self) -> torch.Tensor:
        if self._render_pkg is None:
            return torch.tensor(0.0, device='cuda')
        return self._render_pkg['rgb'].sum() * 0.0

    def distortion_regularization_loss(self) -> torch.Tensor:
        if self._render_pkg is None:
            return torch.tensor(0.0, device='cuda')
        rend_dist = self._render_pkg.get('rend_dist', None)
        if rend_dist is None:
            return self._zero()
        if self._distortion_alpha_mask_min > 0.0:
            rend_alpha = self._render_pkg.get('rend_alpha', None)
            if rend_alpha is not None:
                mask = rend_alpha.detach() > self._distortion_alpha_mask_min
                if bool(mask.any().item()):
                    return rend_dist[mask].mean()
                return self._zero()
        return rend_dist.mean()

    def normal_regularization_loss(self) -> torch.Tensor:
        """Match 2DGS ``train.py``: ``(1 - (rend_normal * surf_normal).sum(0)).mean()``."""
        if self._render_pkg is None:
            return torch.tensor(0.0, device='cuda')
        rend_normal = self._render_pkg.get('rend_normal', None)
        surf_normal = self._render_pkg.get('surf_normal', None)
        if rend_normal is None or surf_normal is None:
            return self._zero()
        normal_error = 1.0 - (rend_normal * surf_normal).sum(dim=0)
        return normal_error.mean()

    def depth_smoothness_regularization_loss(self) -> torch.Tensor:
        """Edge-aware smoothness on ``surf_depth`` (Phase D), guided by GT RGB."""
        if self._render_pkg is None or self._rgb_guide is None:
            return torch.tensor(0.0, device='cuda')
        surf_depth = self._render_pkg.get('surf_depth', None)
        if surf_depth is None:
            return self._zero()
        depth = surf_depth.unsqueeze(0)
        rgb = self._rgb_guide.unsqueeze(0)
        return depth_smoothness_loss(depth, rgb)

    def planar_splat_regularization_loss(self) -> torch.Tensor:
        """No-op for native ``(N, 2)`` surfel storage; kept for config compatibility."""
        raw = self._gaussians.raw_scales
        if raw.shape[1] < 3:
            return self._zero()
        z = raw[:, 2]
        return z.exp().mean()

    def forward(self, render_pkg: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        self._render_pkg = render_pkg
        self._rgb_guide = target
        total = super().forward({
            'L1_Color': {'input': render_pkg['rgb'], 'target': target},
            'DSSIM_Color': {'input': render_pkg['rgb'], 'target': target},
            'DISTORTION_REGULARIZATION': {},
            'NORMAL_REGULARIZATION': {},
            'DEPTH_SMOOTHNESS_REGULARIZATION': {},
            'PLANAR_SPLAT_REGULARIZATION': {},
            'OPACITY_REGULARIZATION': {},
            'SCALE_REGULARIZATION': {},
            'PSNR': {'preds': render_pkg['rgb'], 'target': target, 'data_range': 1.0}
        })
        if self.training:
            with torch.no_grad():
                self._last_geometry_log = {}
                for name in (
                    'DISTORTION_REGULARIZATION',
                    'NORMAL_REGULARIZATION',
                    'DEPTH_SMOOTHNESS_REGULARIZATION',
                ):
                    for metric in self.loss_metrics:
                        if metric.name != name or metric._last_raw is None:
                            continue
                        self._last_geometry_log[name] = {
                            'raw': metric._last_raw,
                            'weighted': metric._last_raw * metric.weight,
                            'weight': float(metric.weight),
                        }
                        break
        return total

    @property
    def last_geometry_log(self) -> dict[str, float]:
        return dict(self._last_geometry_log)
