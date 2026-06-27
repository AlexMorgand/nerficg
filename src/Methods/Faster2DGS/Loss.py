"""Faster2DGS/Loss.py"""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Methods.FasterGS.Model import FasterGSModel
from Optim.Losses.Base import BaseLoss
from Optim.Losses.DSSIM import fused_dssim


class Faster2DGSLoss(BaseLoss):
    """Loss container with 2DGS-style geometric regularization hooks."""

    def __init__(self, loss_config: ConfigParameterList, model: FasterGSModel) -> None:
        super().__init__()
        self._gaussians = model.gaussians
        self._render_pkg: dict[str, torch.Tensor] | None = None
        self._base_distortion_weight = float(loss_config.LAMBDA_DISTORTION)
        self._base_normal_weight = float(loss_config.LAMBDA_NORMAL)
        self._base_planar_splat_weight = float(loss_config.LAMBDA_PLANAR_SPLAT)
        self.add_loss_metric('L1_Color', torch.nn.functional.l1_loss, loss_config.LAMBDA_L1)
        self.add_loss_metric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.add_loss_metric('DISTORTION_REGULARIZATION', self.distortion_regularization_loss, loss_config.LAMBDA_DISTORTION)
        self.add_loss_metric('NORMAL_REGULARIZATION', self.normal_regularization_loss, loss_config.LAMBDA_NORMAL)
        self.add_loss_metric('PLANAR_SPLAT_REGULARIZATION', self.planar_splat_regularization_loss, loss_config.LAMBDA_PLANAR_SPLAT)
        self.add_loss_metric('OPACITY_REGULARIZATION', model.gaussians.opacity_regularization_loss, loss_config.LAMBDA_OPACITY_REGULARIZATION)
        self.add_loss_metric('SCALE_REGULARIZATION', model.gaussians.scale_regularization_loss, loss_config.LAMBDA_SCALE_REGULARIZATION)
        self.add_quality_metric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def set_geometry_loss_weights(
        self,
        *,
        distortion_scale: float = 1.0,
        normal_scale: float = 1.0,
        planar_splat_scale: float = 1.0,
    ) -> None:
        for metric in self.loss_metrics:
            if metric.name == 'DISTORTION_REGULARIZATION':
                metric.weight = max(0.0, self._base_distortion_weight * float(distortion_scale))
            elif metric.name == 'NORMAL_REGULARIZATION':
                metric.weight = max(0.0, self._base_normal_weight * float(normal_scale))
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
        return rend_dist.mean()

    def normal_regularization_loss(self) -> torch.Tensor:
        if self._render_pkg is None:
            return torch.tensor(0.0, device='cuda')
        rend_normal = self._render_pkg.get('rend_normal', None)
        surf_normal = self._render_pkg.get('surf_normal', None)
        rend_alpha = self._render_pkg.get('rend_alpha', None)
        if rend_normal is None or surf_normal is None:
            return self._zero()
        # Training stores n̂_r * alpha and n̂_s * alpha → dot = alpha^2 cos(n̂_r, n̂_s).
        dot = (rend_normal * surf_normal).sum(dim=0)
        if rend_alpha is None:
            return (1.0 - dot).mean()
        alpha = rend_alpha.squeeze(0).clamp_min(0.0)
        cos = dot / (alpha * alpha + 1e-8)
        cos = cos.clamp(-1.0, 1.0)
        mask = alpha > 1e-2
        if not bool(mask.any().item()):
            return self._zero()
        return ((1.0 - cos)[mask]).mean()

    def planar_splat_regularization_loss(self) -> torch.Tensor:
        """Penalize linear scale along local axis 2 (third quaternion column; matches CUDA ``primitive_normal``)."""
        z = self._gaussians.raw_scales[:, 2]
        return z.exp().mean()

    def forward(self, render_pkg: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        self._render_pkg = render_pkg
        rgb = render_pkg['rgb']
        return super().forward({
            'L1_Color': {'input': rgb, 'target': target},
            'DSSIM_Color': {'input': rgb, 'target': target},
            'DISTORTION_REGULARIZATION': {},
            'NORMAL_REGULARIZATION': {},
            'PLANAR_SPLAT_REGULARIZATION': {},
            'OPACITY_REGULARIZATION': {},
            'SCALE_REGULARIZATION': {},
            'PSNR': {'preds': rgb, 'target': target, 'data_range': 1.0}
        })
