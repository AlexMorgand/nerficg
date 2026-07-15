#! /usr/bin/env python3
"""Generate FasterGS-DR benchmark configs mirroring 3DGS-DR train.sh per-scene schedules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / 'configs'
DATA_ROOT = Path('/home/alex/Simulon/dev/3DGS-DR/data')

PROPAGATION_BASE = 8_000
DENSIFY_BASE = 8_000


@dataclass(frozen=True)
class SceneSpec:
    name: str
    data_subdir: str
    base_iterations: int = 61_000
    longer_prop: int = 0
    opac_lr0_interval: int = 200
    white_background: bool = True
    white_background_eval: bool = True
    fields: tuple[str, ...] = ('rgb',)
    densify_extra: int = 0
    dataset_type: str = 'RefNeRFSynthetic'
    test_step: int = 0
    use_env_scope: bool = False
    env_scope_center: tuple[float, float, float] | None = None
    env_scope_radius: float = 0.0


SYNTHETIC_SCENES: tuple[SceneSpec, ...] = (
    SceneSpec('ball', 'ball'),
    SceneSpec('car', 'car'),
    SceneSpec('coffee', 'coffee'),
    SceneSpec('helmet', 'helmet', densify_extra=8_000),
    SceneSpec('teapot', 'teapot', fields=('rgb', 'alpha'), densify_extra=8_000),
    SceneSpec('toaster', 'toaster', longer_prop=24_000),
    SceneSpec('bell', 'bell_blender', base_iterations=91_000, longer_prop=48_000, opac_lr0_interval=0, fields=('rgb', 'alpha')),
    SceneSpec('cat', 'cat_blender', fields=('rgb', 'alpha')),
    SceneSpec('horse', 'horse_blender', longer_prop=36_000, fields=('rgb', 'alpha'), white_background_eval=False),
    SceneSpec('luyu', 'luyu_blender'),
    SceneSpec('potion', 'potion_blender', longer_prop=24_000),
    SceneSpec('tbell', 'tbell_blender', longer_prop=36_000, opac_lr0_interval=0, fields=('rgb', 'alpha')),
    SceneSpec('teapot_glossy', 'teapot_blender', longer_prop=36_000, fields=('rgb', 'alpha')),
)

REAL_SCENES: tuple[SceneSpec, ...] = (
    SceneSpec(
        'gardenspheres',
        'gardenspheres',
        longer_prop=36_000,
        white_background=False,
        dataset_type='Colmap',
        test_step=8,
        use_env_scope=True,
        env_scope_center=(-0.2270, 1.9700, 1.7740),
        env_scope_radius=0.974,
    ),
    SceneSpec(
        'sedan',
        'sedan',
        longer_prop=36_000,
        white_background=False,
        dataset_type='Colmap',
        test_step=8,
        use_env_scope=True,
        env_scope_center=(-0.032, 0.808, 0.751),
        env_scope_radius=2.138,
    ),
    SceneSpec(
        'toycar',
        'toycar',
        longer_prop=36_000,
        white_background=False,
        dataset_type='Colmap',
        test_step=8,
        use_env_scope=True,
        env_scope_center=(0.6810, 0.8080, 4.4550),
        env_scope_radius=2.707,
    ),
)

SCENES: tuple[SceneSpec, ...] = SYNTHETIC_SCENES + REAL_SCENES


def _build_config(spec: SceneSpec) -> dict:
    total_iters = spec.base_iterations + spec.longer_prop + 1
    specular_extra = spec.densify_extra
    densify_end = DENSIFY_BASE + spec.longer_prop + specular_extra
    propagation_end = PROPAGATION_BASE + specular_extra
    bg = [1.0, 1.0, 1.0] if spec.white_background else [0.0, 0.0, 0.0]
    dr_schedule: dict = {
        'INIT_UNTIL_ITERATION': 3_000,
        'PROPAGATION_INTERVAL': 1_000,
        'PROPAGATION_END_ITERATION': propagation_end,
        'LONGER_PROPAGATION_ITERATIONS': spec.longer_prop,
        'PROPAGATION_ENLARGE_SCALE': 1.5,
        'PROPAGATION_MIN_OPACITY': 0.9,
        'PROPAGATION_OPACITY_FLOOR': 0.01,
        'PROPAGATION_MIN_REFLECTION': 0.001,
        'SCALE_ENLARGE_THRESHOLD': 0.02,
        'COLOR_SABOTAGE_THRESHOLD': 0.05,
        'REFLECTION_THRESHOLD': 0.1,
        'COLOR_SABOTAGE_NOISE': 0.4,
        'SPECULAR_TERMINATION_PATIENCE': 0,
        'OPAC_LR0_INTERVAL': spec.opac_lr0_interval,
        'ENVMAP_LEARNING_RATE': 0.05,
        'DENSIFICATION_INTERVAL_DURING_PROPAGATION': 500,
        'PARITY_DIFFUSE_BOOTSTRAP': True,
        'USE_ENV_SCOPE': spec.use_env_scope,
        'ENV_SCOPE_CENTER': list(spec.env_scope_center) if spec.env_scope_center else [0.0, 0.0, 0.0],
        'ENV_SCOPE_RADIUS': spec.env_scope_radius,
        'REFL_MASK_LOSS_WEIGHT': 0.4,
    }
    dataset: dict = {
        'PATH': str(DATA_ROOT / spec.data_subdir),
        'IMAGE_SCALE_FACTOR': 1,
        'NORMALIZE_CUBE': None,
        'NORMALIZE_RECENTER': False,
        'BACKGROUND_COLOR': bg,
        'WHITE_BACKGROUND': spec.white_background,
        'NEAR_PLANE': 0.2,
        'FAR_PLANE': 10000.0,
        'APPLY_PCA': False,
        'APPLY_PCA_RESCALE': False,
    }
    if spec.dataset_type == 'RefNeRFSynthetic':
        dataset['WHITE_BACKGROUND_EVAL'] = spec.white_background_eval
        dataset['POINT_CLOUD_FILE'] = 'points3d.ply'
    else:
        dataset['TEST_STEP'] = spec.test_step
        dataset['IMAGE_FOLDER'] = 'images'
        dataset['SFM_POINTS_FILTER_RATIO'] = 1.0
    return {
        'GLOBAL': {
            'LOG_LEVEL': 2,
            'GPU_INDICES': [0],
            'RANDOM_SEED': 0,
            'ANOMALY_DETECTION': False,
            'FILTER_WARNINGS': True,
            'METHOD_TYPE': 'FasterGS',
            'DATASET_TYPE': spec.dataset_type,
        },
        'MODEL': {
            'SH_DEGREE': 3,
            'PPISP': {'USE': False, 'CONTROLLER_TRAINING_STEPS': 5_000, 'CONTROLLER_DISTILLATION': True},
            'DEFERRED_REFLECTION': {'USE': True, 'REFL_INIT_VALUE': 0.001, 'ENVMAP_RESOLUTION': 128},
        },
        'RENDERER': {
            'SCALE_MODIFIER': 1.0,
            'PROPER_ANTIALIASING': False,
            'FORCE_OPTIMIZED_INFERENCE': False,
        },
        'TRAINING': {
            'LOAD_CHECKPOINT': None,
            'MODEL_NAME': f'fastergs_dr_{spec.name}',
            'NUM_ITERATIONS': total_iters,
            'RUN_VALIDATION': False,
            'DATA': {
                'PRELOADING_LEVEL': 2,
                'FIELDS': list(spec.fields),
                'PRECOMPUTE_RAYS': False,
                'RAYS_TO_DEVICE': False,
            },
            'BACKUP': {
                'FINAL_CHECKPOINT': True,
                'RENDER_TESTSET': True,
                'RENDER_TRAINSET': False,
                'RENDER_VALSET': False,
                'INTERMEDIATE_RENDERINGS': True,
                'VISUALIZE_ERRORS': False,
                'INTERVAL': -1,
                'TRAINING_STATE': False,
            },
            'TIMING': {
                'ACTIVATE': True,
                'INCLUDE_DATALOADING_IN_TOTAL': False,
                'INCLUDE_PRETRAINING_IN_TOTAL': False,
                'INCLUDE_POSTTRAINING_IN_TOTAL': False,
            },
            'WANDB': {
                'ACTIVATE': False,
                'ENTITY': None,
                'PROJECT': 'faster_gs',
                'LOG_IMAGES': True,
                'INDEX_VALIDATION': -1,
                'INDEX_TRAINING': -1,
                'INTERVAL': 500,
                'SWEEP_MODE': {'ACTIVE': False, 'START_ITERATION': 999, 'ITERATION_STRIDE': 1000, 'NUM_IMAGES': -1},
            },
            'WRITE_VRAM_STATS': True,
            'GUI': {
                'ACTIVATE': False,
                'RENDER_INTERVAL': 10,
                'GUI_STATUS_ENABLED': True,
                'GUI_STATUS_INTERVAL': 20,
                'SKIP_GUI_SETUP': False,
                'FPS_ROLLING_AVERAGE_SIZE': 100,
            },
            'DENSIFICATION_START_ITERATION': 500,
            'DENSIFICATION_END_ITERATION': densify_end,
            'DENSIFICATION_INTERVAL': 100,
            'DENSIFICATION_GRAD_THRESHOLD': 0.0002,
            'DENSIFICATION_PERCENT_DENSE': 0.01,
            'SPEEDYSPLAT_PRUNING': {
                'USE': False,
                'START_ITERATION': 6_000,
                'END_ITERATION': spec.base_iterations,
                'INTERVAL': 3_000,
                'SOFT_PRUNING_RATIO': 0.8,
                'HARD_PRUNING_RATIO': 0.3,
            },
            'USE_MCMC': False,
            'MAX_PRIMITIVES': 1_000_000,
            'OPACITY_RESET_INTERVAL': 3_000,
            'EXTRA_OPACITY_RESET_ITERATION': 500,
            'MORTON_ORDERING_INTERVAL': 5_000,
            'MORTON_ORDERING_END_ITERATION': densify_end,
            'FILTER_3D': {'USE': False, 'ORIGINAL_FORMULATION': False, 'FILTER_VARIANCE': 0.2},
            'USE_RANDOM_BACKGROUND_COLOR': False,
            'RANDOM_BACKGROUND_IF_ALPHA_OR_MASK': False,
            'WHITE_BACKGROUND': spec.white_background,
            'MIN_OPACITY_AFTER_TRAINING': 0.00392156862745098,
            'RANDOM_INITIALIZATION': {
                'FORCE': False,
                'N_POINTS': 100_000,
                'ENABLE_CARVING': True,
                'CARVING_IN_ALL_FRUSTUMS': False,
                'CARVING_ENFORCE_ALPHA': False,
            },
            'LOSS': {
                'LAMBDA_L1': 0.8,
                'LAMBDA_DSSIM': 0.2,
                'LAMBDA_OPACITY_REGULARIZATION': 0.0,
                'LAMBDA_SCALE_REGULARIZATION': 0.0,
            },
            'OPTIMIZER': {
                'LEARNING_RATE_MEANS_INIT': 0.00016,
                'LEARNING_RATE_MEANS_FINAL': 1.6e-06,
                'LEARNING_RATE_MEANS_MAX_STEPS': 8_000,
                'LEARNING_RATE_SH_COEFFICIENTS_0': 0.0025,
                'LEARNING_RATE_SH_COEFFICIENTS_REST': 0.000125,
                'LEARNING_RATE_OPACITIES': 0.05,
                'LEARNING_RATE_SCALES': 0.005,
                'LEARNING_RATE_ROTATIONS': 0.001,
                'LEARNING_RATE_REFLECTION_STRENGTH': 0.006,
            },
            'DEFERRED_REFLECTION_SCHEDULE': dr_schedule,
        },
        'DATASET': dataset,
    }


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for spec in SCENES:
        path = CONFIG_DIR / f'fastergs_dr_{spec.name}.yaml'
        cfg = _build_config(spec)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        propagation_end = PROPAGATION_BASE + spec.densify_extra
        effective_prop_end = propagation_end + spec.longer_prop
        densify_end = DENSIFY_BASE + spec.longer_prop + spec.densify_extra
        env_scope = f'env_scope r={spec.env_scope_radius}' if spec.use_env_scope else 'no env_scope'
        print(
            f'wrote {path.name}: loader={spec.dataset_type}, iters={cfg["TRAINING"]["NUM_ITERATIONS"]:,}, '
            f'prop_until={effective_prop_end:,}, densify_until={densify_end:,}, '
            f'opac_lr0={spec.opac_lr0_interval}, densify_extra={spec.densify_extra}, '
            f'white_bg_eval={spec.white_background_eval}, {env_scope}'
        )


if __name__ == '__main__':
    main()
