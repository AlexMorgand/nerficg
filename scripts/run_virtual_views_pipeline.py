#! /usr/bin/env python3

"""End-to-end pipeline: train -> virtual views -> Difix3D -> retrain on augmented COLMAP."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import utils

with utils.DiscoverSourcePath():
    import Framework
    import train
    from Logging import Logger
    from generate_virtual_colmap_views import (
        count_colmap_images,
        default_difix3d_command,
        subdivisions_for_target_views,
    )


NERFICG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIFIX3D_ROOT = NERFICG_ROOT / 'submodules' / 'Difix3D'
FASTERGS_OUTPUT_ROOT = NERFICG_ROOT / 'output' / 'FasterGS'


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _resolve_run_name(dataset_path: Path, total_views: int | None, suffix: str) -> str:
    stem = dataset_path.name.rstrip('/')
    if total_views is not None:
        return f'{stem}_{total_views}{suffix}'
    return f'{stem}{suffix}'


def _stage_training_config(
    base_config_path: Path,
    output_path: Path,
    overrides: dict[str, Any],
) -> Path:
    config = deepcopy(_load_yaml(base_config_path))
    for key, value in overrides.items():
        _set_nested(config, key, value)
    _write_yaml(output_path, config)
    return output_path


def _prepare_augmented_dataset_config(
    base_config_path: Path,
    augmented_dataset_dir: Path,
    model_name: str,
) -> Path:
    config = deepcopy(_load_yaml(base_config_path))
    _set_nested(config, 'DATASET.PATH', str(augmented_dataset_dir.resolve()))
    _set_nested(config, 'TRAINING.EXTERNAL_MASKS_PATH', str((augmented_dataset_dir / 'alpha').resolve()))
    _set_nested(config, 'TRAINING.MODEL_NAME', model_name)
    _set_nested(config, 'TRAINING.GUI.ACTIVATE', False)
    config_path = augmented_dataset_dir / 'gs_generic.yaml'
    _write_yaml(config_path, config)
    return config_path


def _latest_training_output(model_name: str) -> Path:
    matches = sorted(
        FASTERGS_OUTPUT_ROOT.glob(f'{model_name}_*'),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise Framework.TrainingError(f'No training output found for MODEL_NAME={model_name!r} under {FASTERGS_OUTPUT_ROOT}')
    return matches[0]


def _relocate_training_output(source_dir: Path, target_dir: Path) -> Path:
    target_dir = target_dir.resolve()
    source_dir = source_dir.resolve()
    if target_dir == source_dir:
        return target_dir
    if target_dir.exists():
        raise Framework.TrainingError(f'Refusing to overwrite existing training output: {target_dir}')
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_dir), str(target_dir))
    Logger.log_info(f'Renamed training output to {target_dir}')
    return target_dir


def _run_training(config_path: Path, run_name: str | None) -> Path:
    Logger.log_info(f'Starting training with config: {config_path}')
    training_instance = train.main(config_path=str(config_path))
    output_dir = Path(training_instance.output_directory)
    if run_name is None:
        return output_dir
    return _relocate_training_output(output_dir, FASTERGS_OUTPUT_ROOT / run_name)


def _run_generate_virtual_views(
    *,
    initial_run_dir: Path,
    augmented_dataset_dir: Path,
    total_views: int | None,
    subdivisions: int | None,
    pose_mode: str,
    difix3d_command: str | None,
) -> None:
    command = [
        sys.executable,
        str(NERFICG_ROOT / 'scripts' / 'generate_virtual_colmap_views.py'),
        '-d', str(initial_run_dir),
        '-o', str(augmented_dataset_dir),
        '--pose-mode', pose_mode,
    ]
    if subdivisions is not None:
        command.extend(['--subdivisions', str(subdivisions)])
    elif total_views is not None:
        command.extend(['--total-views', str(total_views)])
    if difix3d_command is not None:
        command.extend(['--difix3d-command', difix3d_command])
    Logger.log_info(f'Running: {" ".join(command)}')
    subprocess.run(command, check=True, cwd=str(NERFICG_ROOT))


def _ensure_difix3d_root(path: Path) -> Path:
    resolved = path.resolve()
    script = resolved / 'sample_test_video.py'
    if not script.is_file():
        raise Framework.TrainingError(
            f'Difix3D not found at {resolved}. '
            f'Initialize the submodule with: git submodule update --init submodules/Difix3D'
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='run_virtual_views_pipeline.py',
        description=(
            'Run the full sparse-view augmentation pipeline:\n'
            '  1) initial FasterGS training\n'
            '  2) virtual COLMAP view generation\n'
            '  3) optional Difix3D restoration\n'
            '  4) final FasterGS training on the augmented dataset'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Source COLMAP dataset directory (contains sparse/, images/, gs_generic.yaml).',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Training YAML for step 1 (default: <dataset>/gs_generic.yaml).',
    )
    parser.add_argument(
        '--total-views',
        type=int,
        required=True,
        help='Target total number of views in the augmented dataset.',
    )
    parser.add_argument(
        '--initial-run-name',
        type=str,
        default=None,
        help='Name for the first FasterGS output directory under output/FasterGS/.',
    )
    parser.add_argument(
        '--augmented-dataset-name',
        type=str,
        default=None,
        help='Name for the generated COLMAP dataset under output/FasterGS/.',
    )
    parser.add_argument(
        '--final-run-name',
        type=str,
        default=None,
        help='Name for the final FasterGS output directory under output/FasterGS/.',
    )
    parser.add_argument(
        '--initial-model-name',
        type=str,
        default=None,
        help='TRAINING.MODEL_NAME for step 1 (default: derived from initial run name).',
    )
    parser.add_argument(
        '--final-model-name',
        type=str,
        default=None,
        help='TRAINING.MODEL_NAME for step 4 (default: derived from final run name).',
    )
    parser.add_argument(
        '--pose-mode',
        choices=['sphere_gaps', 'sequence'],
        default='sphere_gaps',
        help='Virtual camera placement strategy (default: sphere_gaps).',
    )
    parser.add_argument(
        '--subdivisions',
        type=int,
        default=None,
        help='Override automatic subdivision calculation for virtual view generation.',
    )
    parser.add_argument(
        '--difix3d-root',
        type=Path,
        default=DEFAULT_DIFIX3D_ROOT,
        help='Path to the Difix3D repository (default: submodules/Difix3D).',
    )
    parser.add_argument(
        '--skip-difix3d',
        action='store_true',
        help='Skip Difix3D restoration and keep raw GS renders for virtual views.',
    )
    parser.add_argument(
        '--skip-initial-train',
        action='store_true',
        help='Skip step 1 and reuse --initial-run-dir.',
    )
    parser.add_argument(
        '--initial-run-dir',
        type=Path,
        default=None,
        help='Existing initial FasterGS output directory (required with --skip-initial-train).',
    )
    parser.add_argument(
        '--skip-virtual-views',
        action='store_true',
        help='Skip step 2 and reuse an existing augmented dataset directory.',
    )
    parser.add_argument(
        '--skip-final-train',
        action='store_true',
        help='Stop after virtual view generation / Difix3D.',
    )
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_dir():
        raise Framework.TrainingError(f'Dataset directory not found: {dataset_path}')

    base_config_path = (args.config or (dataset_path / 'gs_generic.yaml')).expanduser().resolve()
    if not base_config_path.is_file():
        raise Framework.TrainingError(f'Training config not found: {base_config_path}')

    n_original = count_colmap_images(dataset_path)
    subdivisions, actual_total = subdivisions_for_target_views(n_original, args.total_views, args.pose_mode)
    if args.subdivisions is not None:
        subdivisions = args.subdivisions
        Logger.log_info(f'Using manual --subdivisions={subdivisions}')
    else:
        Logger.log_info(
            f'{n_original} original views -> target {args.total_views} views '
            f'via subdivisions={subdivisions} (actual total={actual_total}).'
        )

    initial_run_name = args.initial_run_name or _resolve_run_name(dataset_path, n_original, '')
    augmented_dataset_name = args.augmented_dataset_name or _resolve_run_name(dataset_path, args.total_views, '')
    final_run_name = args.final_run_name or f'{augmented_dataset_name}_full'

    initial_model_name = args.initial_model_name or initial_run_name
    final_model_name = args.final_model_name or final_run_name

    initial_run_dir = args.initial_run_dir.expanduser().resolve() if args.initial_run_dir else FASTERGS_OUTPUT_ROOT / initial_run_name
    augmented_dataset_dir = FASTERGS_OUTPUT_ROOT / augmented_dataset_name

    Logger.set_mode(Logger.MODE_VERBOSE)

    if args.skip_initial_train:
        if not initial_run_dir.is_dir():
            raise Framework.TrainingError(f'--skip-initial-train requires existing directory: {initial_run_dir}')
        Logger.log_info(f'Reusing initial training output: {initial_run_dir}')
    else:
        staged_config = _stage_training_config(
            base_config_path=base_config_path,
            output_path=FASTERGS_OUTPUT_ROOT / f'{initial_run_name}_config.yaml',
            overrides={
                'DATASET.PATH': str(dataset_path),
                'TRAINING.MODEL_NAME': initial_model_name,
                'TRAINING.GUI.ACTIVATE': False,
            },
        )
        initial_run_dir = _run_training(staged_config, initial_run_name)

    difix3d_command = None
    if not args.skip_difix3d:
        difix3d_root = _ensure_difix3d_root(args.difix3d_root)
        difix3d_command = default_difix3d_command(difix3d_root)

    if args.skip_virtual_views:
        if not augmented_dataset_dir.is_dir():
            raise Framework.TrainingError(f'--skip-virtual-views requires existing directory: {augmented_dataset_dir}')
        Logger.log_info(f'Reusing augmented dataset: {augmented_dataset_dir}')
    else:
        _run_generate_virtual_views(
            initial_run_dir=initial_run_dir,
            augmented_dataset_dir=augmented_dataset_dir,
            total_views=None if args.subdivisions is not None else args.total_views,
            subdivisions=args.subdivisions,
            pose_mode=args.pose_mode,
            difix3d_command=difix3d_command,
        )

    final_config_path = _prepare_augmented_dataset_config(
        base_config_path=base_config_path,
        augmented_dataset_dir=augmented_dataset_dir,
        model_name=final_model_name,
    )
    Logger.log_info(f'Wrote augmented dataset config: {final_config_path}')

    if args.skip_final_train:
        Logger.log_info('Pipeline finished (final training skipped).')
        return

    final_run_dir = _run_training(final_config_path, final_run_name)
    Logger.log_info('Pipeline finished.')
    Logger.log_info(f'Initial run: {initial_run_dir}')
    Logger.log_info(f'Augmented dataset: {augmented_dataset_dir}')
    Logger.log_info(f'Final run: {final_run_dir}')


if __name__ == '__main__':
    main()
