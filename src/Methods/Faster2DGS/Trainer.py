"""Faster2DGS/Trainer.py"""

import torch

import Framework
from Datasets.Base import BaseDataset
from Datasets.utils import apply_background_color, get_supervision_alpha
from Logging import Logger
from Methods.Base.utils import pre_training_callback, training_callback, post_training_callback
from Methods.Faster2DGS.Faster2DGSCudaBackend import SurfelAuxMode
from Methods.Faster2DGS.Loss import Faster2DGSLoss
from Methods.FasterGS.Trainer import FasterGSTrainer


@Framework.Configurable.configure(
    # Official 2DGS train.py: lambda_dist when iteration > 3000, lambda_normal when iteration > 7000.
    DISTORTION_START_ITERATION=3_000,
    NORMAL_START_ITERATION=7_000,
    DEPTH_SMOOTHNESS_START_ITERATION=7_000,
    PLANAR_SPLAT_START_ITERATION=7_000,
    GEOMETRY_LOG_INTERVAL=10,
    # Official densify_from_iter=500 but first densify at 600 (iteration > densify_from_iter).
    DENSIFICATION_START_ITERATION=500,
    DENSIFICATION_OPACITY_CULL=0.05,
    DENSIFICATION_MAX_SCREEN_SIZE=20,
    SKIP_FINAL_OPACITY_PRUNE=True,
    OPACITY_RESET_MAX=0.01,
    # Optional splat budget (off by default).
    USE_SPLAT_BUDGET=False,
    SPLAT_BUDGET_TARGET=0,
    SPLAT_BUDGET_HARD_CAP=0,
    SPLAT_BUDGET_START_ITERATION=1_500,
    SPLAT_BUDGET_GRAD_SCALE_MAX=2.0,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
        LAMBDA_DISTORTION=0.0,
        DISTORTION_ALPHA_MASK_MIN=0.0,
        LAMBDA_NORMAL=0.05,
        LAMBDA_DEPTH_SMOOTHNESS=0.0,
        LAMBDA_PLANAR_SPLAT=0.0,
        LAMBDA_OPACITY_REGULARIZATION=0.0,
        LAMBDA_SCALE_REGULARIZATION=0.0,
    ),
)
class Faster2DGSTrainer(FasterGSTrainer):
    """Trainer with 2DGS regularization hooks matching official train.py loop order."""

    def _geometry_active(self, iteration: int, start_iteration: int) -> bool:
        """Match official ``train.py``: loss weight is zero until ``iteration > start``."""
        return iteration > start_iteration

    def _distortion_scale(self, iteration: int) -> float:
        return 1.0 if self._geometry_active(iteration, self.DISTORTION_START_ITERATION) else 0.0

    def _normal_scale(self, iteration: int) -> float:
        return 1.0 if self._geometry_active(iteration, self.NORMAL_START_ITERATION) else 0.0

    def _depth_smoothness_scale(self, iteration: int) -> float:
        return 1.0 if self._geometry_active(iteration, self.DEPTH_SMOOTHNESS_START_ITERATION) else 0.0

    def _planar_splat_scale(self, iteration: int) -> float:
        return 1.0 if self._geometry_active(iteration, self.PLANAR_SPLAT_START_ITERATION) else 0.0

    @pre_training_callback(priority=40)
    @torch.no_grad()
    def setup_gaussians(self, _, dataset: 'BaseDataset') -> None:
        """Initialize Gaussians, then attach 2DGS loss (FasterGSTrainer.setup_gaussians creates FasterGSLoss)."""
        super().setup_gaussians(_, dataset)
        if hasattr(self.model.gaussians, 'opacity_reset_max'):
            self.model.gaussians.opacity_reset_max = float(self.OPACITY_RESET_MAX)
        self.loss = Faster2DGSLoss(loss_config=self.LOSS, model=self.model)

    def _needs_aux_maps(self, iteration: int) -> bool:
        if self._distortion_scale(iteration) > 0.0:
            return True
        if self._normal_scale(iteration) > 0.0:
            return True
        if self._depth_smoothness_scale(iteration) > 0.0:
            return True
        if self._planar_splat_scale(iteration) > 0.0:
            return True
        return False

    def _aux_mode(self, iteration: int) -> int:
        if self._normal_scale(iteration) > 0.0 or self._depth_smoothness_scale(iteration) > 0.0:
            return SurfelAuxMode.FULL
        if self._distortion_scale(iteration) > 0.0:
            return SurfelAuxMode.DISTORTION
        return SurfelAuxMode.PHOTOMETRIC

    def _needs_surf_normal(self, iteration: int) -> bool:
        """``depth_to_normal`` is only required when normal or depth-smoothness losses are active."""
        return self._normal_scale(iteration) > 0.0 or self._depth_smoothness_scale(iteration) > 0.0

    def _select_bg_color(self, view) -> torch.Tensor:
        if self.USE_RANDOM_BACKGROUND_COLOR:
            return torch.rand_like(view.camera.background_color)
        if getattr(self, 'RANDOM_BACKGROUND_IF_ALPHA_OR_MASK', False) and get_supervision_alpha(view) is not None:
            return torch.rand_like(view.camera.background_color)
        return view.camera.background_color

    def _should_densify(self, iteration: int) -> bool:
        return (
            not self.USE_MCMC
            and iteration > self.DENSIFICATION_START_ITERATION
            and iteration < self.DENSIFICATION_END_ITERATION
            and iteration % self.DENSIFICATION_INTERVAL == 0
        )

    def _should_reset_opacities(self, iteration: int, dataset: 'BaseDataset') -> bool:
        if self.USE_MCMC:
            return False
        if iteration % self.OPACITY_RESET_INTERVAL == 0:
            return True
        return (
            iteration == self.EXTRA_OPACITY_RESET_ITERATION
            and dataset.default_camera.background_color.sum() != 0.0
        )

    def _effective_densification_grad_threshold(self, iteration: int) -> float:
        """Optionally raise grad bar above budget target (disabled when USE_SPLAT_BUDGET=False)."""
        base = float(self.DENSIFICATION_GRAD_THRESHOLD)
        if not self.USE_SPLAT_BUDGET:
            return base
        target = int(self.SPLAT_BUDGET_TARGET)
        if target <= 0 or iteration < self.SPLAT_BUDGET_START_ITERATION:
            return base
        n = self.model.gaussians.means.shape[0]
        if n <= target:
            return base
        ratio = min(float(self.SPLAT_BUDGET_GRAD_SCALE_MAX), (n / float(target)) ** 0.5)
        return base * ratio

    def _enforce_splat_budget(self, iteration: int) -> int:
        if not self.USE_SPLAT_BUDGET:
            return 0
        hard_cap = int(self.SPLAT_BUDGET_HARD_CAP)
        if hard_cap <= 0 or iteration < self.SPLAT_BUDGET_START_ITERATION:
            return 0
        g = self.model.gaussians
        if not hasattr(g, 'prune_to_budget') or g.means.shape[0] <= hard_cap:
            return 0
        return g.prune_to_budget(hard_cap)

    @training_callback(priority=100, active=False)
    @torch.no_grad()
    def densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Disabled — densify runs after backward inside ``training_iteration`` (official 2DGS order)."""

    @training_callback(priority=90, active=False)
    @torch.no_grad()
    def reset_opacities(self, *_) -> None:
        """Disabled — opacity reset runs after densify inside ``training_iteration``."""

    @training_callback(priority=89, active=False)
    @torch.no_grad()
    def reset_opacities_extra(self, _, dataset: 'BaseDataset') -> None:
        """Disabled — merged into ``_should_reset_opacities``."""

    def _densify_grace_after_opacity_reset(self, iteration: int) -> bool:
        """Skip aggressive prune on the first densify after each opacity reset.

        Reset caps all opacities at 0.01 (< opacity_cull 0.05). One densify later,
        screen-size pruning with stale max_radii2D was removing ~50% of kitchen splats
        before opacities could recover (iter 3100/6100).
        """
        if iteration <= self.OPACITY_RESET_INTERVAL:
            return False
        return iteration % self.OPACITY_RESET_INTERVAL == 100

    @torch.no_grad()
    def _run_densify(self, iteration: int, dataset: 'BaseDataset') -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.USE_MCMC:
            self.model.gaussians.mcmc_densification(min_opacity=0.005, cap_max=self.MAX_PRIMITIVES)
        else:
            grace = self._densify_grace_after_opacity_reset(iteration)
            max_screen_size = (
                None
                if grace
                else (
                    self.DENSIFICATION_MAX_SCREEN_SIZE
                    if iteration > self.OPACITY_RESET_INTERVAL
                    else None
                )
            )
            if self.requires_empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model.gaussians.adaptive_density_control(
                self._effective_densification_grad_threshold(iteration),
                self.DENSIFICATION_OPACITY_CULL,
                iteration > self.OPACITY_RESET_INTERVAL and not grace,
                max_screen_size=max_screen_size,
                skip_opacity_prune=grace,
            )
            budget_pruned = self._enforce_splat_budget(iteration)
            stats = getattr(self.model.gaussians, '_last_densify_stats', None)
            breakdown = getattr(self.model.gaussians, '_last_prune_breakdown', None)
            if stats is not None:
                msg = (
                    f'iter {iteration} densify: '
                    f'{stats["n_before"]:,} -> {stats["n_after"]:,} '
                    f'(grad={stats["grad_candidates"]:,}, dup={stats["n_duplicate"]:,}, '
                    f'split={stats["n_split_parents"]:,}, new={stats["n_new"]:,}, pruned={stats["n_pruned"]:,})'
                )
                if breakdown is not None:
                    msg += (
                        f' [prune: split={breakdown["split_parents"]:,}, '
                        f'opacity={breakdown["opacity"]:,}, screen={breakdown["screen_size"]:,}, '
                        f'world={breakdown["world_scale"]:,}, rot={breakdown["rotation"]:,}]'
                    )
                if budget_pruned > 0:
                    msg += f' [budget: -{budget_pruned:,} -> {self.model.gaussians.means.shape[0]:,}]'
                Logger.log_info(msg)
            if iteration < self.DENSIFICATION_END_ITERATION:
                self.model.gaussians.reset_densification_info()
        if self.requires_empty_cache:
            torch.cuda.empty_cache()
        if self.FILTER_3D.USE:
            self.model.gaussians.compute_3d_filter(dataset.train())

    @training_callback(priority=80)
    def training_iteration(self, iteration: int, dataset: 'BaseDataset') -> None:
        """Official 2DGS order: backward → densify → opacity reset → optimizer."""
        self.model.train()
        dataset.train()
        self.loss.train()
        self.model.gaussians.update_learning_rate(iteration + 1)
        self.loss.set_geometry_loss_weights(
            distortion_scale=self._distortion_scale(iteration),
            normal_scale=self._normal_scale(iteration),
            depth_smoothness_scale=self._depth_smoothness_scale(iteration),
            planar_splat_scale=self._planar_splat_scale(iteration),
        )
        view = self.train_sampler.get(dataset=dataset)['view']
        bg_color = self._select_bg_color(view)
        render_pkg = self.renderer.render_image_training(
            view=view,
            update_densification_info=not self.USE_MCMC and iteration < self.DENSIFICATION_END_ITERATION,
            bg_color=bg_color,
            aux_mode=self._aux_mode(iteration),
            compute_surf_normal=self._needs_surf_normal(iteration),
        )
        rgb_gt = view.rgb
        if (supervision_alpha := get_supervision_alpha(view)) is not None:
            rgb_gt = apply_background_color(rgb_gt, supervision_alpha, bg_color)
        loss = self.loss(render_pkg, rgb_gt)
        loss.backward()
        if (
            not self.USE_MCMC
            and iteration < self.DENSIFICATION_END_ITERATION
            and (radii := render_pkg.get('radii')) is not None
            and radii.numel() > 0
            and hasattr(self.model.gaussians, 'update_max_radii2D')
        ):
            with torch.no_grad():
                self.model.gaussians.update_max_radii2D(radii)
        if self._should_densify(iteration):
            self._run_densify(iteration, dataset)
        if self._should_reset_opacities(iteration, dataset):
            if iteration == self.EXTRA_OPACITY_RESET_ITERATION and dataset.default_camera.background_color.sum() != 0.0:
                Logger.log_info('resetting opacities one additional time because using non-black background')
            self.model.gaussians.reset_opacities()
        self.model.gaussians.optimizer.step()
        self.model.gaussians.optimizer.zero_grad(set_to_none=True)
        self.model.gaussians.post_optimizer_step(inject_noise=self.USE_MCMC)
        if self.model.ppisp is not None:
            self.model.ppisp.step()
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
            with torch.no_grad():
                mean_opacity = float(self.model.gaussians.opacities.mean().item())
            Logger.log_info(f'iter {iteration} geometry: ' + ', '.join(parts) + f', mean_opacity={mean_opacity:.4f}')

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
