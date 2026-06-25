#! /usr/bin/env python3

"""Generate intermediate COLMAP views from a trained FasterGS splat."""

import shutil
import subprocess
import sys
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import numpy as np
import pycolmap
import torch

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Cameras.utils import invert_3d_affine
    from Datasets.utils import View, save_image
    from Implementations import Datasets as DI
    from Implementations import Methods as MI
    from Logging import Logger
    from Visual.Trajectories.InterpolatedPath import _interpolate_c2w


def count_colmap_images(dataset_path: Path) -> int:
    """Returns the number of registered COLMAP images."""
    images_txt = dataset_path / 'sparse' / '0' / 'images.txt'
    if images_txt.is_file():
        count = 0
        for line in images_txt.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if stripped.split()[0].isdigit() and len(stripped.split()) >= 10:
                count += 1
        if count > 0:
            return count
    images_dir = dataset_path / 'images'
    if images_dir.is_dir():
        return len([p for p in images_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}])
    raise Framework.InferenceError(f'Cannot count COLMAP images under {dataset_path}')


def subdivisions_for_target_views(n_original: int, target_total: int, pose_mode: str) -> tuple[int, int]:
    """Returns (subdivisions, resulting_total_views) for the requested pose mode."""
    if n_original < 2:
        raise Framework.InferenceError('Need at least two reference views to generate virtual views.')
    if target_total < n_original:
        raise Framework.InferenceError(
            f'--total-views ({target_total}) must be >= the number of reference views ({n_original}).'
        )
    if pose_mode == 'sphere_gaps':
        subdivisions = max(2, (target_total + n_original - 1) // n_original)
        actual_total = n_original * subdivisions
    elif pose_mode == 'sequence':
        extra = target_total - n_original
        segment_count = n_original - 1
        subdivisions = max(2, 1 + (extra + segment_count - 1) // segment_count)
        actual_total = n_original + segment_count * (subdivisions - 1)
    else:
        raise Framework.InferenceError(f'Unsupported pose mode for --total-views: {pose_mode}')
    return subdivisions, actual_total


def default_difix3d_command(difix3d_root: Path) -> str:
    """Returns a command template that runs the bundled Difix3D sample script."""
    script = difix3d_root / 'sample_test_video.py'
    if not script.is_file():
        raise Framework.InferenceError(f'Difix3D script not found: {script}')
    return (
        f'cd {difix3d_root} && XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 '
        f'{sys.executable} sample_test_video.py {{input_dir}} {{output_dir}}'
    )


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        raise Framework.InferenceError('Cannot normalize near-zero vector while generating virtual camera poses.')
    return v / norm


def _view_source_c2w(view: View) -> np.ndarray:
    """Returns the nerficg training-space moving-camera pose for this view."""
    original_c2w = view.exif.get('turntable_original_c2w')
    if original_c2w is not None:
        return np.asarray(original_c2w, dtype=np.float64)
    return view.c2w_numpy


def _view_export_c2w(view: View) -> np.ndarray:
    """Returns the raw COLMAP-space pose that should be written to images.txt."""
    export_c2w = view.exif.get('colmap_export_c2w')
    if export_c2w is not None:
        return np.asarray(export_c2w, dtype=np.float64)
    return _view_source_c2w(view)


def _raw_colmap_views_by_name(dataset_path: Path) -> dict[str, tuple[np.ndarray, int, int]]:
    reconstruction = pycolmap.Reconstruction(dataset_path / 'sparse' / '0')
    return {
        image.name: (image.cam_from_world().inverse().matrix().astype(np.float64), int(image.camera.camera_id), int(image.image_id))
        for image in reconstruction.images.values()
    }


def _raw_colmap_point_center(dataset_path: Path, fallback_positions: np.ndarray) -> np.ndarray:
    reconstruction = pycolmap.Reconstruction(dataset_path / 'sparse' / '0')
    points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float64)
    if points.size == 0:
        return fallback_positions.mean(axis=0)
    return np.median(points, axis=0)


def _attach_raw_colmap_export_poses(reference_views: list[View], dataset_path: Path) -> None:
    raw_views = _raw_colmap_views_by_name(dataset_path)
    for view in reference_views:
        if view._rgb is None:
            continue
        name = view._rgb.path.name
        if name in raw_views:
            export_c2w, camera_id, image_id = raw_views[name]
            view.exif['colmap_export_c2w'] = export_c2w
            view.exif['colmap_camera_id'] = camera_id
            view.exif['colmap_sort_key'] = float(image_id)


def _group_reference_views(reference_views: list[View], group_by_camera: bool) -> list[list[View]]:
    if not group_by_camera:
        return [sorted(reference_views, key=lambda view: (view.timestamp, view.global_frame_idx))]
    groups: dict[int, list[View]] = {}
    for view in reference_views:
        groups.setdefault(view.camera_index, []).append(view)
    return [
        sorted(group, key=lambda view: (view.timestamp, view.frame_idx, view.global_frame_idx))
        for _, group in sorted(groups.items())
    ]


def _training_from_export_transform(reference_views: list[View]) -> np.ndarray:
    for view in reference_views:
        return _view_source_c2w(view) @ invert_3d_affine(_view_export_c2w(view))
    return np.eye(4, dtype=np.float64)


def _fibonacci_sphere_directions(n: int) -> np.ndarray:
    indices = np.arange(n, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / float(n)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * golden_angle
    return np.stack([np.cos(theta) * radius, np.sin(theta) * radius, z], axis=1)


def _look_at_preserve_roll(eye: np.ndarray, center: np.ndarray, reference_c2w: np.ndarray) -> np.ndarray:
    forward = _normalize(center - eye)
    right_hint = reference_c2w[:3, 0]
    right = right_hint - np.dot(right_hint, forward) * forward
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    right = _normalize(right)
    down = _normalize(np.cross(forward, right))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def _select_gap_directions(existing_directions: np.ndarray, n_virtual: int, candidate_count: int) -> np.ndarray:
    candidates = _fibonacci_sphere_directions(max(candidate_count, n_virtual * 8, 256))
    selected: list[np.ndarray] = []
    occupied = existing_directions.copy()
    for _ in range(n_virtual):
        similarities = candidates @ occupied.T
        min_angular_distance = np.arccos(np.clip(similarities.max(axis=1), -1.0, 1.0))
        best_idx = int(np.argmax(min_angular_distance))
        best = candidates[best_idx]
        selected.append(best)
        occupied = np.vstack([occupied, best])
    return np.asarray(selected, dtype=np.float64)


def _make_sequence_virtual_views(reference_views: list[View], subdivisions: int, group_by_camera: bool) -> list[tuple[View, str]]:
    if subdivisions < 2:
        raise Framework.InferenceError('--subdivisions must be >= 2; use 2 to add one midpoint per pose pair.')

    virtual_views: list[tuple[View, str]] = []
    for group in _group_reference_views(reference_views, group_by_camera):
        for segment_idx, (left, right) in enumerate(zip(group[:-1], group[1:])):
            c0 = _view_source_c2w(left)
            c1 = _view_source_c2w(right)
            export_c0 = _view_export_c2w(left)
            export_c1 = _view_export_c2w(right)
            left_name = left._rgb.path.stem if left._rgb is not None else f'{left.global_frame_idx:05d}'
            right_name = right._rgb.path.stem if right._rgb is not None else f'{right.global_frame_idx:05d}'
            for sub_idx in range(1, subdivisions):
                t = sub_idx / float(subdivisions)
                source_c2w = _interpolate_c2w(c0, c1, t)
                export_c2w = _interpolate_c2w(export_c0, export_c1, t)
                ref = left if t < 0.5 else right
                exif = {
                    'colmap_camera_id': _camera_id_for_view(ref),
                    'colmap_export_c2w': export_c2w,
                    'colmap_sort_key': (1.0 - t) * float(left.exif.get('colmap_sort_key', left.global_frame_idx)) + t * float(right.exif.get('colmap_sort_key', right.global_frame_idx)),
                }
                name = f'{left_name}__interp_{sub_idx:02d}_of_{subdivisions:02d}__{right_name}.png'
                virtual_views.append((
                    View(
                        camera=deepcopy(ref.camera),
                        camera_index=ref.camera_index,
                        frame_idx=segment_idx * (subdivisions - 1) + sub_idx - 1,
                        global_frame_idx=len(virtual_views),
                        c2w=source_c2w,
                        timestamp=(1.0 - t) * left.timestamp + t * right.timestamp,
                        exif=exif,
                    ),
                    name,
                ))
    return virtual_views


def _make_sphere_gap_virtual_views(
    reference_views: list[View],
    subdivisions: int,
    group_by_camera: bool,
    dataset_path: Path,
    candidate_count: int,
) -> list[tuple[View, str]]:
    if subdivisions < 2:
        raise Framework.InferenceError('--subdivisions must be >= 2; use 2 to add roughly one full extra shell of views.')

    training_from_export = _training_from_export_transform(reference_views)
    all_export_positions = np.stack([_view_export_c2w(view)[:3, 3] for view in reference_views])
    center = _raw_colmap_point_center(dataset_path, all_export_positions)
    virtual_views: list[tuple[View, str]] = []

    for group_idx, group in enumerate(_group_reference_views(reference_views, group_by_camera)):
        export_c2ws = [_view_export_c2w(view) for view in group]
        positions = np.stack([c2w[:3, 3] for c2w in export_c2ws])
        radii = np.linalg.norm(positions - center[None], axis=1)
        radius = float(np.median(radii))
        directions = np.stack([_normalize(position - center) for position in positions])
        n_virtual = len(group) * (subdivisions - 1)
        gap_directions = _select_gap_directions(directions, n_virtual, candidate_count)

        for local_idx, direction in enumerate(gap_directions):
            eye = center + direction * radius
            nearest_idx = int(np.argmax(directions @ direction))
            ref = group[nearest_idx]
            export_c2w = _look_at_preserve_roll(eye, center, export_c2ws[nearest_idx])
            source_c2w = training_from_export @ export_c2w
            exif = {
                'colmap_camera_id': _camera_id_for_view(ref),
                'colmap_export_c2w': export_c2w,
                'colmap_sort_key': float(ref.exif.get('colmap_sort_key', ref.global_frame_idx)) + (local_idx + 1) / float(n_virtual + 1),
            }
            camera_suffix = f'_cam{_camera_id_for_view(ref):02d}' if group_by_camera else ''
            name = f'sphere_gap{camera_suffix}_{local_idx:05d}.png'
            virtual_views.append((
                View(
                    camera=deepcopy(ref.camera),
                    camera_index=ref.camera_index,
                    frame_idx=local_idx,
                    global_frame_idx=len(virtual_views),
                    c2w=source_c2w,
                    timestamp=ref.timestamp,
                    exif=exif,
                ),
                name,
            ))
    return virtual_views


def _make_virtual_views(
    reference_views: list[View],
    subdivisions: int,
    group_by_camera: bool,
    pose_mode: str,
    dataset_path: Path,
    candidate_count: int,
) -> list[tuple[View, str]]:
    if pose_mode == 'sequence':
        return _make_sequence_virtual_views(reference_views, subdivisions, group_by_camera)
    if pose_mode == 'sphere_gaps':
        return _make_sphere_gap_virtual_views(reference_views, subdivisions, group_by_camera, dataset_path, candidate_count)
    raise Framework.InferenceError(f'Unknown pose mode: {pose_mode}')


def _has_turntable_views(reference_views: list[View]) -> bool:
    return any('turntable_fixed_c2w' in view.exif and 'turntable_original_c2w' in view.exif for view in reference_views)


def _camera_id_for_view(view: View) -> int:
    return int(view.exif.get('colmap_camera_id', view.camera_index + 1))


def _camera_lines(views: list[tuple[View, str]]) -> list[str]:
    cameras: dict[int, object] = {}
    for view, _ in views:
        cameras.setdefault(_camera_id_for_view(view), view.camera)
    lines = [
        '# Camera list with one line of data per camera:',
        '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]',
        f'# Number of cameras: {len(cameras)}',
    ]
    for camera_id, camera in sorted(cameras.items()):
        lines.append(
            f'{camera_id} PINHOLE {int(camera.width)} {int(camera.height)} '
            f'{float(camera.focal_x):.10f} {float(camera.focal_y):.10f} '
            f'{float(camera.center_x):.10f} {float(camera.center_y):.10f}'
        )
    return lines


def _read_raw_camera_lines(source_dataset_path: Path) -> list[str]:
    cameras_path = source_dataset_path / 'sparse' / '0' / 'cameras.txt'
    if not cameras_path.exists():
        return []
    return [line for line in cameras_path.read_text(encoding='utf-8').splitlines() if line]


def _rotmat_to_colmap_qvec(R: np.ndarray) -> np.ndarray:
    """Converts a world-to-camera rotation matrix to COLMAP qvec order (qw, qx, qy, qz)."""
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0.0, 0.0, 0.0],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0.0, 0.0],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0.0],
        [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
    ], dtype=np.float64) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    qvec[1:] *= -1
    return qvec


def _qvec_tvec_from_c2w(source_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w2c = invert_3d_affine(source_c2w)
    qvec = _rotmat_to_colmap_qvec(w2c[:3, :3])
    return qvec.astype(np.float64), w2c[:3, 3].astype(np.float64)


def _image_lines(views: list[tuple[View, str]]) -> list[str]:
    lines = [
        '# Image list with two lines of data per image:',
        '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME',
        '#   POINTS2D[] as (X, Y, POINT3D_ID)',
        f'# Number of images: {len(views)}',
    ]
    sorted_views = sorted(views, key=lambda item: (float(item[0].exif.get('colmap_sort_key', item[0].global_frame_idx)), item[1]))
    for image_id, (view, name) in enumerate(sorted_views, start=1):
        qvec, tvec = _qvec_tvec_from_c2w(_view_export_c2w(view))
        camera_id = _camera_id_for_view(view)
        lines.append(
            f'{image_id} '
            f'{qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} '
            f'{tvec[0]:.10f} {tvec[1]:.10f} {tvec[2]:.10f} '
            f'{camera_id} {name}'
        )
        lines.append('')
    return lines


def _write_colmap_sparse(output_dir: Path, views: list[tuple[View, str]], source_dataset_path: Path) -> None:
    sparse_dir = output_dir / 'sparse' / '0'
    sparse_dir.mkdir(parents=True, exist_ok=True)
    raw_camera_lines = _read_raw_camera_lines(source_dataset_path)
    if raw_camera_lines:
        (sparse_dir / 'cameras.txt').write_text('\n'.join(raw_camera_lines) + '\n', encoding='utf-8')
    else:
        (sparse_dir / 'cameras.txt').write_text('\n'.join(_camera_lines(views)) + '\n', encoding='utf-8')
    (sparse_dir / 'images.txt').write_text('\n'.join(_image_lines(views)) + '\n', encoding='utf-8')
    source_points3d = source_dataset_path / 'sparse' / '0' / 'points3D.txt'
    if source_points3d.exists():
        shutil.copy2(source_points3d, sparse_dir / 'points3D.txt')
    else:
        (sparse_dir / 'points3D.txt').write_text(
            '# 3D point list with one line of data per point:\n'
            '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
            '# Number of points: 0, mean track length: 0\n',
            encoding='utf-8',
        )


def _copy_original_image(view: View, output_path: Path) -> None:
    if view._rgb is None:
        raise Framework.InferenceError(f'Cannot copy original image for view {view.global_frame_idx}: missing rgb path.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(view._rgb.path, output_path)


def _run_difix3d(command_template: str, input_dir: Path, output_dir: Path, alpha_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = command_template.format(
        input_dir=input_dir,
        output_dir=output_dir,
        alpha_dir=alpha_dir,
    )
    Logger.log_info(f'Running Difix3D command: {command}')
    subprocess.run(command, shell=True, check=True)


def main(
    *,
    base_dir: Path,
    output_dir: Path,
    checkpoint_name: str,
    reference_split: str,
    subdivisions: int,
    pose_mode: str,
    sphere_candidate_count: int,
    group_by_camera: bool,
    virtual_only: bool,
    difix3d_command: str | None,
) -> None:
    Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
    try:
        dataset = DI.get_dataset(
            dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
            path=Framework.config.DATASET.PATH,
        )
        if reference_split not in dataset.data or not dataset.data[reference_split]:
            raise Framework.InferenceError(f'Invalid or empty reference split: {reference_split}')

        model = MI.get_model(
            method=Framework.config.GLOBAL.METHOD_TYPE,
            checkpoint=str(base_dir / 'checkpoints' / checkpoint_name),
        ).eval()
        renderer = MI.get_renderer(
            method=Framework.config.GLOBAL.METHOD_TYPE,
            model=model,
        )

        reference_views = list(dataset.data[reference_split])
        _attach_raw_colmap_export_poses(reference_views, Path(Framework.config.DATASET.PATH))
        is_turntable = _has_turntable_views(reference_views)
        if is_turntable:
            Logger.log_info(
                'detected turntable metadata: interpolating the original COLMAP moving-camera poses. '
                'If the generated dataset is later loaded with TURNTABLE.ENABLED, nerficg will reinterpret '
                'those poses as fixed-camera renders with rotating Gaussians.'
            )
        virtual_views = _make_virtual_views(
            reference_views,
            subdivisions,
            group_by_camera,
            pose_mode,
            Path(Framework.config.DATASET.PATH),
            sphere_candidate_count,
        )
        if not virtual_views:
            raise Framework.InferenceError('No virtual views generated; need at least two reference views per camera group.')

        images_dir = output_dir / 'images'
        alpha_dir = output_dir / 'alpha'
        virtual_raw_dir = output_dir / 'images_nerficg_virtual'
        virtual_restored_dir = output_dir / 'images_nerficg_virtual_fixed'
        images_dir.mkdir(parents=True, exist_ok=True)
        alpha_dir.mkdir(parents=True, exist_ok=True)
        virtual_raw_dir.mkdir(parents=True, exist_ok=True)

        output_views: list[tuple[View, str]] = []
        if not virtual_only:
            for view in reference_views:
                name = view._rgb.path.name if view._rgb is not None else f'original_{view.global_frame_idx:05d}.png'
                output_views.append((view, name))
                _copy_original_image(view, images_dir / name)
                alpha = renderer.render_alpha_inference(view, to_chw=True)
                save_image(alpha_dir / name, alpha)

        for view, name in Logger.log_progress(virtual_views, desc='rendering virtual views', leave=False, total=len(virtual_views)):
            rgb = renderer.render_image(view, to_chw=True)['rgb']
            alpha = renderer.render_alpha_inference(view, to_chw=True)
            save_image(virtual_raw_dir / name, rgb)
            save_image(images_dir / name, rgb)
            save_image(alpha_dir / name, alpha)
            output_views.append((view, name))

        if difix3d_command is not None:
            _run_difix3d(difix3d_command, virtual_raw_dir, virtual_restored_dir, alpha_dir)
            for _, name in virtual_views:
                restored = virtual_restored_dir / name
                if not restored.exists():
                    Logger.log_warning(f'Difix3D did not produce {restored}; keeping raw GS render in images/.')
                    continue
                shutil.copy2(restored, images_dir / name)

        _write_colmap_sparse(output_dir, output_views, Path(Framework.config.DATASET.PATH))
        Logger.log_info(f'Wrote {len(virtual_views)} virtual view(s) to {output_dir}')
        Logger.log_info(f'COLMAP sparse output: {output_dir / "sparse" / "0"}')
    finally:
        Framework.teardown()


if __name__ == '__main__':
    parser = ArgumentParser(description='Render intermediate virtual COLMAP views from a trained FasterGS splat.')
    parser.add_argument('-d', '--dir', dest='base_dir', type=Path, required=True, help='Training output directory.')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=Path, required=True, help='Output dataset directory.')
    parser.add_argument('--checkpoint', dest='checkpoint_name', default='final.pt', help='Checkpoint filename.')
    parser.add_argument('--reference-split', dest='reference_split', default='train', help='Dataset split to interpolate.')
    parser.add_argument('--subdivisions', dest='subdivisions', type=int, default=None, help='sequence: segment subdivisions; sphere_gaps: multiplier for new views, 2 roughly doubles views.')
    parser.add_argument(
        '--total-views',
        dest='total_views',
        type=int,
        default=None,
        help='Target total view count (original + virtual). Computes --subdivisions automatically.',
    )
    parser.add_argument(
        '--pose-mode',
        dest='pose_mode',
        choices=['sphere_gaps', 'sequence'],
        default='sphere_gaps',
        help='How to place virtual cameras. sphere_gaps fills missing directions on the viewing sphere; sequence interpolates adjacent file-order poses.',
    )
    parser.add_argument(
        '--sphere-candidate-count',
        dest='sphere_candidate_count',
        type=int,
        default=4096,
        help='Number of Fibonacci candidates used by --pose-mode sphere_gaps.',
    )
    parser.add_argument('--no-group-by-camera', dest='group_by_camera', action='store_false', help='Interpolate one global stream instead of per camera.')
    parser.add_argument('--virtual-only', dest='virtual_only', action='store_true', help='Do not copy original images into the output dataset.')
    parser.add_argument(
        '--difix3d-command',
        dest='difix3d_command',
        default=None,
        help='Optional command template. Placeholders: {input_dir}, {output_dir}, {alpha_dir}.',
    )
    parser.set_defaults(group_by_camera=True)
    args = parser.parse_args()
    if args.subdivisions is not None and args.total_views is not None:
        raise Framework.InferenceError('Use either --subdivisions or --total-views, not both.')
    subdivisions = args.subdivisions
    if subdivisions is None:
        if args.total_views is None:
            subdivisions = 2
        else:
            import yaml
            training_cfg = yaml.safe_load(open(args.base_dir / 'training_config.yaml', encoding='utf-8'))
            dataset_path = Path(training_cfg['DATASET']['PATH'])
            n_original = count_colmap_images(dataset_path)
            subdivisions, actual_total = subdivisions_for_target_views(n_original, args.total_views, args.pose_mode)
            if actual_total != args.total_views:
                Logger.log_warning(
                    f'--total-views={args.total_views} resolves to {actual_total} views '
                    f'with --subdivisions={subdivisions} and pose_mode={args.pose_mode}.'
                )
            else:
                Logger.log_info(
                    f'--total-views={args.total_views} -> --subdivisions={subdivisions} '
                    f'({n_original} original + {actual_total - n_original} virtual).'
                )
    main_kwargs = vars(args)
    main_kwargs['subdivisions'] = subdivisions
    main_kwargs.pop('total_views', None)
    main(**main_kwargs)
