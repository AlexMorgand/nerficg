#! /usr/bin/env python3

"""fastergs_eval_gate.py: Validate masked-eval inputs and turntable train/infer parity."""

from argparse import ArgumentParser
from pathlib import Path

import torch

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger
    from Datasets.utils import list_sorted_directories, list_sorted_files
    from Implementations import Datasets as DI
    from Implementations import Methods as MI


def _image_names(path: Path) -> list[str]:
    return [
        name for name in list_sorted_files(path)
        if name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg')
    ]


def _assert_identical(scene: str, kind: str, expected: list[str], got: list[str]) -> None:
    if expected == got:
        return
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    details = []
    if missing:
        details.append(f'missing={missing[:5]}')
    if extra:
        details.append(f'extra={extra[:5]}')
    raise Framework.InferenceError(
        f'scene "{scene}" {kind} mismatch: expected {len(expected)} files, got {len(got)} ({", ".join(details)})'
    )


def validate_masked_eval_inputs(results_root: Path, scenes: list[str] | None = None) -> None:
    if not results_root.is_dir():
        raise Framework.InferenceError(f'invalid --results-root: "{results_root}"')
    scene_names = scenes or [
        s for s in list_sorted_directories(results_root)
        if (results_root / s / 'gt').is_dir()
    ]
    if not scene_names:
        raise Framework.InferenceError(f'no scenes with gt/ found in "{results_root}"')

    for scene in scene_names:
        scene_path = results_root / scene
        gt_path = scene_path / 'gt'
        if not gt_path.is_dir():
            raise Framework.InferenceError(f'scene "{scene}" missing gt/ directory')
        gt_names = _image_names(gt_path)
        if not gt_names:
            raise Framework.InferenceError(f'scene "{scene}" has no GT images in gt/')

        mask_path = scene_path / '_mask'
        if mask_path.exists():
            mask_names = _image_names(mask_path)
            _assert_identical(scene, 'mask filenames', gt_names, mask_names)

        method_names = [
            m for m in list_sorted_directories(scene_path)
            if m not in {'gt', '_mask'} and not m.startswith('_')
        ]
        if not method_names:
            Logger.log_warning(f'scene "{scene}" has no method directories')
            continue
        for method_name in method_names:
            method_names_list = _image_names(scene_path / method_name)
            _assert_identical(scene, f'method "{method_name}" filenames', gt_names, method_names_list)
        Logger.log_info(f'scene "{scene}": validated {len(method_names)} methods with {len(gt_names)} aligned images')


@torch.no_grad()
def validate_turntable_parity(
    training_dir: Path,
    checkpoint_name: str,
    subset: str,
    max_views: int,
    mae_threshold: float,
    max_threshold: float,
) -> None:
    config_path = training_dir / 'training_config.yaml'
    checkpoint_path = training_dir / 'checkpoints' / checkpoint_name
    if not config_path.is_file():
        raise Framework.InferenceError(f'missing training config: "{config_path}"')
    if not checkpoint_path.is_file():
        raise Framework.InferenceError(f'missing checkpoint: "{checkpoint_path}"')

    Framework.setup(config_path=str(config_path), require_custom_config=True)
    try:
        dataset = DI.get_dataset(
            dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
            path=Framework.config.DATASET.PATH
        )
        if not getattr(Framework.config.DATASET, 'TURNTABLE', None) or not Framework.config.DATASET.TURNTABLE.ENABLED:
            Logger.log_warning('turntable mode is disabled in dataset config; parity check skipped')
            return

        model = MI.get_model(
            method=Framework.config.GLOBAL.METHOD_TYPE,
            checkpoint=str(checkpoint_path),
        ).eval()
        renderer = MI.get_renderer(
            method=Framework.config.GLOBAL.METHOD_TYPE,
            model=model
        )
        if subset not in dataset.data or not dataset.data[subset]:
            raise Framework.InferenceError(f'invalid subset "{subset}" for parity check')
        dataset.set_mode(subset)
        views = list(dataset)
        if max_views > 0 and len(views) > max_views:
            stride = max(1, len(views) // max_views)
            views = views[::stride][:max_views]

        maes: list[float] = []
        maxes: list[float] = []
        for idx, view in enumerate(views):
            img_fast = renderer.render_image_training(
                view=view,
                update_densification_info=False,
                bg_color=view.camera.background_color,
            ).clamp(0.0, 1.0)
            img_infer = renderer.render_image_inference(view, to_chw=True)['rgb']
            err = (img_fast - img_infer).abs()
            mae = float(err.mean().item())
            mxe = float(err.max().item())
            maes.append(mae)
            maxes.append(mxe)
            Logger.log_info(f'parity view {idx:03d}: mae={mae:.6f}, max={mxe:.6f}')

        mean_mae = sum(maes) / max(1, len(maes))
        peak = max(maxes) if maxes else 0.0
        Logger.log_info(
            f'turntable parity summary ({len(maes)} views): '
            f'mean_mae={mean_mae:.6f}, max_abs={peak:.6f}, thresholds=({mae_threshold:.6f}, {max_threshold:.6f})'
        )
        if mean_mae > mae_threshold or peak > max_threshold:
            raise Framework.InferenceError(
                f'turntable parity failed: mean_mae={mean_mae:.6f}, max_abs={peak:.6f}'
            )
    finally:
        Framework.teardown()


def main() -> None:
    parser = ArgumentParser(
        prog='fastergs_eval_gate.py',
        description='Validate masked-eval filename alignment and turntable train/infer render parity.'
    )
    parser.add_argument('--results-root', type=str, default=None, help='Root directory containing scenes with gt/, methods/, and optional _mask/.')
    parser.add_argument('--scenes', type=str, nargs='*', default=None, help='Optional scene names to validate under --results-root.')
    parser.add_argument('--training-dir', type=str, default=None, help='Training output directory containing training_config.yaml and checkpoints/.')
    parser.add_argument('--checkpoint', type=str, default='final.pt', help='Checkpoint filename inside training-dir/checkpoints/.')
    parser.add_argument('--subset', type=str, default='test', help='Subset to sample for turntable parity check.')
    parser.add_argument('--max-views', type=int, default=8, help='Max subset views to test for parity.')
    parser.add_argument('--mae-threshold', type=float, default=1e-2, help='Fail if mean MAE exceeds this threshold.')
    parser.add_argument('--max-threshold', type=float, default=8e-2, help='Fail if any pixel abs error exceeds this threshold.')
    args, _ = parser.parse_known_args()

    Logger.set_mode(Logger.MODE_VERBOSE)
    if args.results_root is None and args.training_dir is None:
        raise Framework.InferenceError('nothing to validate: pass --results-root and/or --training-dir')

    if args.results_root is not None:
        validate_masked_eval_inputs(Path(args.results_root), args.scenes)
    if args.training_dir is not None:
        validate_turntable_parity(
            training_dir=Path(args.training_dir),
            checkpoint_name=args.checkpoint,
            subset=args.subset,
            max_views=max(1, args.max_views),
            mae_threshold=max(0.0, args.mae_threshold),
            max_threshold=max(0.0, args.max_threshold),
        )
    Logger.log_info('evaluation gate checks passed')


if __name__ == '__main__':
    main()
