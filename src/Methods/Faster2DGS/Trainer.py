"""Faster2DGS/Trainer.py"""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import apply_background_color, get_supervision_alpha
from Logging import Logger
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.Faster2DGS.Loss import Faster2DGSLoss
from Methods.FasterGS.Trainer import FasterGSTrainer


@Framework.Configurable.configure(
    DISTORTION_START_ITERATION=3_000,
    NORMAL_START_ITERATION=7_000,
    DEPTH_SMOOTHNESS_START_ITERATION=7_000,
    PLANAR_SPLAT_START_ITERATION=7_000,
    GEOMETRY_WARMUP_DURATION=0,  # 0 = hard enable at start iteration, >0 = linear ramp to full weight
    GEOMETRY_LOG_INTERVAL=10,
    # Official 2DGS OptimizationParams.opacity_cull (was hardcoded 0.005 in FasterGSTrainer).
    DENSIFICATION_OPACITY_CULL=0.05,
    DENSIFICATION_MAX_SCREEN_SIZE=20,
    SKIP_FINAL_OPACITY_PRUNE=True,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
        # 2DGS paper: α=100 (unbounded) / 1000 (bounded) for L_d, β=0.05 for L_n; code defaults 0 — set in yaml.
        LAMBDA_DISTORTION=0.0,
        LAMBDA_NORMAL=0.05,
        LAMBDA_DEPTH_SMOOTHNESS=0.0,
        LAMBDA_PLANAR_SPLAT=0.0,
        LAMBDA_OPACITY_REGULARIZATION=0.0,
        LAMBDA_SCALE_REGULARIZATION=0.0,
    ),
)
class Faster2DGSTrainer(FasterGSTrainer):
    """Trainer with 2DGS regularization hooks."""

    def _warmup_scale(self, iteration: int, start_iteration: int) -> float:
        if iteration <= start_iteration:
            return 0.0
        if self.GEOMETRY_WARMUP_DURATION <= 0:
            return 1.0
        return min(1.0, (iteration - start_iteration) / float(self.GEOMETRY_WARMUP_DURATION))

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        """Initialize Gaussians, then attach 2DGS loss (FasterGSTrainer.setup_gaussians creates FasterGSLoss)."""
        super().setup_gaussians(_, dataset)
        self.loss = Faster2DGSLoss(loss_config=self.LOSS, model=self.model)

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        self.model.train()
        dataset.train()
        self.loss.train()
        self.model.gaussians.update_learning_rate(iteration + 1)
        self.loss.set_geometry_loss_weights(
            distortion_scale=self._warmup_scale(iteration, self.DISTORTION_START_ITERATION),
            normal_scale=self._warmup_scale(iteration, self.NORMAL_START_ITERATION),
            depth_smoothness_scale=self._warmup_scale(iteration, self.DEPTH_SMOOTHNESS_START_ITERATION),
            planar_splat_scale=self._warmup_scale(iteration, self.PLANAR_SPLAT_START_ITERATION),
        )
        view = self.train_sampler.get(dataset=dataset)['view']
        bg_color = torch.rand_like(view.camera.background_color) if self.USE_RANDOM_BACKGROUND_COLOR else view.camera.background_color
        render_pkg = self.renderer.render_image_training(
            view=view,
            update_densification_info=not self.USE_MCMC and iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
        )
        rgb_gt = view.rgb
        if (supervision_alpha := get_supervision_alpha(view)) is not None:
            rgb_gt = apply_background_color(rgb_gt, supervision_alpha, bg_color)
        loss = self.loss(render_pkg, rgb_gt)
        loss.backward()
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad(set_to_none=True)
        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
        if (
            not self.USE_MCMC
            and iteration < self.DENSIFICATION_END_ITERATION
            and (radii := render_pkg.get('radii')) is not None
            and radii.numel() > 0
            and hasattr(self.model.gaussians, 'update_max_radii2D')
        ):
            with torch.no_grad():
                self.model.gaussians.update_max_radii2D(radii)
        if (
            self.GEOMETRY_LOG_INTERVAL > 0
            and iteration % self.GEOMETRY_LOG_INTERVAL == 0
            and hasattr(self.loss, 'last_geometry_log')
            and self.loss.last_geometry_log
        ):
            parts = []
            for key, stats in self.loss.last_geometry_log.items():
                short = key.replace('_REGULARIZATION', '').lower()
                parts.append(f'{short}={stats["weighted"]:.5f}(w={stats["weight"]:.4g}, raw={stats["raw"]:.5f})')
            Logger.log_info(f'iter {iteration} geometry: ' + ', '.join(parts))

    @training_callback(priority=100, start_iteration='DENSIFICATION_START_ITERATION', end_iteration='DENSIFICATION_END_ITERATION', iteration_stride='DENSIFICATION_INTERVAL')
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """2DGS-aligned densification — runs before train step so param resize never races backward."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.USE_MCMC:
            self.model.gaussians.mcmc_densification(min_opacity=0.005, cap_max=self.MAX_PRIMITIVES)
        else:
            max_screen_size = (
                self.DENSIFICATION_MAX_SCREEN_SIZE
                if iteration > self.OPACITY_RESET_INTERVAL
                else None
            )
            if self.requires_empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model.gaussians.adaptive_density_control(
                self.DENSIFICATION_GRAD_THRESHOLD,
                self.DENSIFICATION_OPACITY_CULL,
                iteration > self.OPACITY_RESET_INTERVAL,
                max_screen_size=max_screen_size,
            )
            stats = getattr(self.model.gaussians, '_last_densify_stats', None)
            if stats is not None:
                Logger.log_info(
                    f'iter {iteration} densify: '
                    f'{stats["n_before"]:,} -> {stats["n_after"]:,} '
                    f'(grad={stats["grad_candidates"]:,}, dup={stats["n_duplicate"]:,}, '
                    f'split={stats["n_split_parents"]:,}, new={stats["n_new"]:,}, pruned={stats["n_pruned"]:,})'
                )
            if iteration < self.DENSIFICATION_END_ITERATION:
                self.model.gaussians.reset_densification_info()
        if self.requires_empty_cache:
            torch.cuda.empty_cache()
        if self.FILTER_3D.USE:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, _, dataset: 'BaseDataset') -> None:
        """Clean up after training (official 2DGS has no final opacity prune)."""
        if self.SKIP_FINAL_OPACITY_PRUNE:
            prune_mask = self.model.gaussians._rotations.mul(self.model.gaussians._rotations).sum(dim=1) < 1e-8
            if bool(prune_mask.any().item()):
                self.model.gaussians.prune(prune_mask)
            n_gaussians = self.model.gaussians.means.shape[0]
            self.model.gaussians._densification_info = None
            self.model.gaussians._filter_3d = None
            self.model.gaussians.apply_morton_ordering()
        else:
            n_gaussians = self.model.gaussians.training_cleanup(min_opacity=self.MIN_OPACITY_AFTER_TRAINING)
        Logger.log_info(f'final number of Gaussians: {n_gaussians:,}')
        with open(str(self.output_directory / 'n_gaussians.txt'), 'w') as n_gaussians_file:
            n_gaussians_file.write(
                f'Final number of Gaussians: {n_gaussians:,}\n'
                f'\n'
                f'N_Gaussians:{n_gaussians}'
            )
        if self.model.ppisp is not None and self.model.ppisp.config.controller_distillation:
            Logger.log_info('distilling PPISP controller')
            with torch.enable_grad():
                self.model.train()
                dataset.train()
                self.loss.train()
                for _ in Logger.log_progress(range(self.model.ppisp.config.controller_training_steps)):
                    view = self.train_sampler.get(dataset=dataset)['view']
                    image = self.renderer.ppisp_controller_distillation(view=view)
                    rgb_gt = view.rgb
                    if (supervision_alpha := get_supervision_alpha(view)) is not None:
                        rgb_gt = apply_background_color(rgb_gt, supervision_alpha, view.camera.background_color)
                    loss = self.loss(image, rgb_gt)
                    loss.backward()
                    self.model.ppisp.step()
            self.model.ppisp.create_report(self.output_directory)
