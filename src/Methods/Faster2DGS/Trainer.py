"""Faster2DGS/Trainer.py"""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import apply_background_color, check_external_mask_tensor
from Methods.Base.utils import training_callback
from Methods.Faster2DGS.Loss import Faster2DGSLoss
from Methods.FasterGS.Trainer import FasterGSTrainer


@Framework.Configurable.configure(
    DISTORTION_START_ITERATION=3_000,
    NORMAL_START_ITERATION=7_000,
    PLANAR_SPLAT_START_ITERATION=7_000,
    GEOMETRY_WARMUP_DURATION=0,  # 0 = hard enable at start iteration, >0 = linear ramp to full weight
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
        LAMBDA_DISTORTION=0.0,
        LAMBDA_NORMAL=0.0,
        # Push splats toward thin disks in the local XY plane (small scale along local +Z / aux normal axis).
        LAMBDA_PLANAR_SPLAT=0.0,
        LAMBDA_OPACITY_REGULARIZATION=0.0,
        LAMBDA_SCALE_REGULARIZATION=0.0,
    ),
)
class Faster2DGSTrainer(FasterGSTrainer):
    """Trainer with 2DGS regularization hooks."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss = Faster2DGSLoss(loss_config=self.LOSS, gaussians=self.model.gaussians)

    def _warmup_scale(self, iteration: int, start_iteration: int) -> float:
        if iteration <= start_iteration:
            return 0.0
        if self.GEOMETRY_WARMUP_DURATION <= 0:
            return 1.0
        return min(1.0, (iteration - start_iteration) / float(self.GEOMETRY_WARMUP_DURATION))

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        self.model.train()
        dataset.train()
        self.loss.train()
        self.model.gaussians.update_learning_rate(iteration + 1)
        self.loss.set_geometry_loss_weights(
            distortion_scale=self._warmup_scale(iteration, self.DISTORTION_START_ITERATION),
            normal_scale=self._warmup_scale(iteration, self.NORMAL_START_ITERATION),
            planar_splat_scale=self._warmup_scale(iteration, self.PLANAR_SPLAT_START_ITERATION),
        )
        view = self.train_sampler.get(dataset=dataset)['view']
        external_mask = None
        if self.EXTERNAL_MASKS_PATH not in (None, ''):
            external_mask = view.segmentation
            if external_mask is None:
                raise Framework.TrainingError('external masks enabled but current view has no segmentation mask')
            label = str(getattr(getattr(view, '_rgb', None), 'path', 'unknown'))
            external_mask = check_external_mask_tensor(external_mask, view.rgb, label)
        alpha_gt = view.alpha
        supervision_alpha = alpha_gt
        if external_mask is not None:
            supervision_alpha = external_mask if supervision_alpha is None else supervision_alpha * external_mask
        use_random_bg = self.USE_RANDOM_BACKGROUND_COLOR or (
            self.RANDOM_BACKGROUND_IF_ALPHA_OR_MASK and supervision_alpha is not None
        )
        bg_color = torch.rand_like(view.camera.background_color) if use_random_bg else view.camera.background_color
        render_pkg = self.renderer.render_image_training(
            view=view,
            update_densification_info=not self.USE_MCMC and iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
        )
        rgb_gt = view.rgb
        if supervision_alpha is not None:
            rgb_gt = apply_background_color(rgb_gt, supervision_alpha, bg_color)
        loss = self.loss(render_pkg, rgb_gt)
        loss.backward()
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad()
        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
