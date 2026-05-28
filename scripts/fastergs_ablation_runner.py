#! /usr/bin/env python3

"""fastergs_ablation_runner.py: Build/run FasterGS ablation and stress-test config sets."""

from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VariantSpec:
    name: str
    overrides: dict[str, Any]
    description: str


ABLATION_VARIANTS: list[VariantSpec] = [
    VariantSpec(
        name='mesh_locked',
        description='Mesh prior locked geometry, no densification.',
        overrides={
            'TRAINING.MODEL_NAME': 'ablate_mesh_locked',
            'TRAINING.FREEZE_INITIAL_GAUSSIAN_GEOMETRY': True,
            'TRAINING.ENABLE_DENSIFICATION_WITH_GAUSSIAN_PLY_INIT': False,
            'TRAINING.RANDOM_INITIALIZATION.IGNORE_GAUSSIAN_PLY': False,
            'TRAINING.USE_MCMC': False,
        },
    ),
    VariantSpec(
        name='mesh_selective_unfreeze',
        description='Mesh prior with trainable geometry, no densification.',
        overrides={
            'TRAINING.MODEL_NAME': 'ablate_mesh_unfreeze',
            'TRAINING.FREEZE_INITIAL_GAUSSIAN_GEOMETRY': False,
            'TRAINING.ENABLE_DENSIFICATION_WITH_GAUSSIAN_PLY_INIT': False,
            'TRAINING.RANDOM_INITIALIZATION.IGNORE_GAUSSIAN_PLY': False,
            'TRAINING.USE_MCMC': False,
        },
    ),
    VariantSpec(
        name='mesh_unfreeze_densify',
        description='Mesh prior with trainable geometry and slow densification.',
        overrides={
            'TRAINING.MODEL_NAME': 'ablate_mesh_unfreeze_densify',
            'TRAINING.FREEZE_INITIAL_GAUSSIAN_GEOMETRY': False,
            'TRAINING.ENABLE_DENSIFICATION_WITH_GAUSSIAN_PLY_INIT': True,
            'TRAINING.DENSIFICATION_GRAD_THRESHOLD_MULTIPLIER_GAUSSIAN_PLY_INIT': 2.0,
            'TRAINING.DENSIFICATION_INTERVAL_MULTIPLIER_GAUSSIAN_PLY_INIT': 2,
            'TRAINING.RANDOM_INITIALIZATION.IGNORE_GAUSSIAN_PLY': False,
            'TRAINING.USE_MCMC': False,
        },
    ),
    VariantSpec(
        name='pointcloud_adc',
        description='COLMAP point cloud init with classic ADC-like densification.',
        overrides={
            'TRAINING.MODEL_NAME': 'ablate_pointcloud_adc',
            'TRAINING.RANDOM_INITIALIZATION.IGNORE_GAUSSIAN_PLY': True,
            'TRAINING.USE_MCMC': False,
            'TRAINING.DENSIFICATION_END_ITERATION': 14900,
        },
    ),
    VariantSpec(
        name='pointcloud_mcmc',
        description='COLMAP point cloud init with MCMC-style bounded growth.',
        overrides={
            'TRAINING.MODEL_NAME': 'ablate_pointcloud_mcmc',
            'TRAINING.RANDOM_INITIALIZATION.IGNORE_GAUSSIAN_PLY': True,
            'TRAINING.USE_MCMC': True,
            'TRAINING.DENSIFICATION_END_ITERATION': 24900,
            'TRAINING.MORTON_ORDERING_END_ITERATION': 25000,
            'TRAINING.LOSS.LAMBDA_OPACITY_REGULARIZATION': 0.01,
            'TRAINING.LOSS.LAMBDA_SCALE_REGULARIZATION': 0.01,
            'TRAINING.OPTIMIZER.LEARNING_RATE_OPACITIES': 0.05,
        },
    ),
    VariantSpec(
        name='mesh_mcmc_unfreeze',
        description='Mesh prior with trainable geometry and MCMC densification.',
        overrides={
            'TRAINING.MODEL_NAME': 'ablate_mesh_mcmc',
            'TRAINING.FREEZE_INITIAL_GAUSSIAN_GEOMETRY': False,
            'TRAINING.ENABLE_DENSIFICATION_WITH_GAUSSIAN_PLY_INIT': True,
            'TRAINING.RANDOM_INITIALIZATION.IGNORE_GAUSSIAN_PLY': False,
            'TRAINING.USE_MCMC': True,
            'TRAINING.DENSIFICATION_END_ITERATION': 24900,
            'TRAINING.MORTON_ORDERING_END_ITERATION': 25000,
            'TRAINING.LOSS.LAMBDA_OPACITY_REGULARIZATION': 0.01,
            'TRAINING.LOSS.LAMBDA_SCALE_REGULARIZATION': 0.01,
            'TRAINING.OPTIMIZER.LEARNING_RATE_OPACITIES': 0.05,
        },
    ),
]


STRESS_VARIANTS: list[VariantSpec] = [
    VariantSpec(
        name='stress_reduced_views_x2',
        description='Reduce training-view density with TEST_STEP=2 (~every second frame held out).',
        overrides={
            'TRAINING.MODEL_NAME': 'stress_views_x2',
            'DATASET.TEST_STEP': 2,
        },
    ),
    VariantSpec(
        name='stress_reduced_views_x3',
        description='Aggressive reduced-view regime with TEST_STEP=3.',
        overrides={
            'TRAINING.MODEL_NAME': 'stress_views_x3',
            'DATASET.TEST_STEP': 3,
        },
    ),
    VariantSpec(
        name='stress_reference_mid',
        description='Turntable fixed camera at mid sequence for frame sensitivity.',
        overrides={
            'TRAINING.MODEL_NAME': 'stress_ref_mid',
            'DATASET.TURNTABLE.REFERENCE_FRAME_IDX': 32,
        },
    ),
]


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _write_variant_config(base_config: dict[str, Any], spec: VariantSpec, output_dir: Path, short_iters: int | None) -> Path:
    cfg = deepcopy(base_config)
    for key, value in spec.overrides.items():
        _set_nested(cfg, key, value)
    if short_iters is not None:
        _set_nested(cfg, 'TRAINING.NUM_ITERATIONS', int(short_iters))
        _set_nested(cfg, 'TRAINING.MORTON_ORDERING_END_ITERATION', int(short_iters))
        _set_nested(cfg, 'TRAINING.DENSIFICATION_END_ITERATION', min(int(short_iters) - 100, int(cfg['TRAINING'].get('DENSIFICATION_END_ITERATION', short_iters))))
    _set_nested(cfg, 'TRAINING.BACKUP.RENDER_TESTSET', True)
    _set_nested(cfg, 'TRAINING.BACKUP.INTERMEDIATE_RENDERINGS', False)
    _set_nested(cfg, 'TRAINING.BACKUP.VISUALIZE_ERRORS', False)
    _set_nested(cfg, 'TRAINING.WANDB.ACTIVATE', False)
    _set_nested(cfg, 'TRAINING.GUI.ACTIVATE', False)
    cfg_path = output_dir / f'{spec.name}.yaml'
    with open(cfg_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def _train_process(config_path: str, output_queue: Queue) -> None:
    import utils
    with utils.DiscoverSourcePath():
        import train
    instance = train.main(config_path=config_path)
    output_queue.put(str(instance.output_directory))


def _run_train(config_path: Path) -> str:
    q: Queue = Queue()
    p = Process(target=_train_process, args=(str(config_path), q))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f'training failed for {config_path} (exit_code={p.exitcode})')
    return q.get(block=False)


def _build_variants(base_config_path: Path, output_dir: Path, specs: list[VariantSpec], short_iters: int | None) -> list[Path]:
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = [
        _write_variant_config(base_cfg, spec, output_dir, short_iters)
        for spec in specs
    ]
    manifest = {
        'base_config': str(base_config_path),
        'variants': [
            {'name': spec.name, 'description': spec.description, 'config': str(path)}
            for spec, path in zip(specs, config_paths)
        ],
    }
    with open(output_dir / 'manifest.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return config_paths


def main() -> None:
    parser = ArgumentParser(
        prog='fastergs_ablation_runner.py',
        description='Generate and optionally run FasterGS ablation/stress config sets.'
    )
    parser.add_argument('--base-config', type=str, required=True, help='Base training YAML to branch from.')
    parser.add_argument('--output-dir', type=str, required=True, help='Where generated config variants are written.')
    parser.add_argument('--mode', type=str, default='ablation', choices=['ablation', 'stress'], help='Which variant set to generate.')
    parser.add_argument('--short-iters', type=int, default=12000, help='Override NUM_ITERATIONS for quick sprint comparisons.')
    parser.add_argument('--run', action='store_true', help='Run training for each generated config sequentially.')
    args, _ = parser.parse_known_args()

    base_config_path = Path(args.base_config)
    output_dir = Path(args.output_dir)
    specs = ABLATION_VARIANTS if args.mode == 'ablation' else STRESS_VARIANTS
    config_paths = _build_variants(
        base_config_path=base_config_path,
        output_dir=output_dir,
        specs=specs,
        short_iters=max(1, args.short_iters) if args.short_iters > 0 else None,
    )
    print(f'generated {len(config_paths)} {args.mode} configs in "{output_dir}"')

    if not args.run:
        return

    run_log_path = output_dir / 'run_outputs.txt'
    with open(run_log_path, 'w', encoding='utf-8') as run_log:
        for cfg in config_paths:
            print(f'running training for "{cfg.name}"')
            output_dir_run = _run_train(cfg)
            run_log.write(f'{cfg}\t{output_dir_run}\n')
            run_log.flush()
    print(f'completed all runs; outputs listed in "{run_log_path}"')


if __name__ == '__main__':
    main()
