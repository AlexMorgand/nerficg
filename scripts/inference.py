#! /usr/bin/env python3

"""inference.py: Renders outputs from a pretrained model."""

import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import torch

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Logging import Logger
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    from Visual.Trajectories import CameraTrajectory


def _encode_rgb_pngs_to_mp4(*, rgb_dir: Path, output_mp4: Path, fps: int, num_frames: int, video_codec: str) -> None:
    """Encodes a numbered rgb/*.png sequence (00000.png …) to H.264 MP4 using ffmpeg."""
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise Framework.InferenceError(
            'ffmpeg was not found in PATH; required for --interpolate-video / --orbit-video.'
        )
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        '-y',
        '-hide_banner',
        '-loglevel',
        'error',
        '-framerate',
        str(fps),
        '-start_number',
        '0',
        '-i',
        str(rgb_dir / '%05d.png'),
        '-frames:v',
        str(num_frames),
        '-c:v',
        video_codec,
        '-pix_fmt',
        'yuv420p',
        str(output_mp4),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or '').strip()
        raise Framework.InferenceError(f'ffmpeg failed encoding video: {stderr or e}') from e


def _trajectory_reference_set(subset: str, interpolate_reference_split: str, orbit_reference_split: str) -> str | None:
    """Reference split for trajectory keyframes when adding to dataset."""
    if subset == 'interpolated_path':
        return interpolate_reference_split
    if subset == 'pose_ordered_orbit':
        return orbit_reference_split
    if subset == 'recorded_path':
        return None
    return 'train'


def _render_trajectory_to_mp4(
    *,
    renderer,
    dataset,
    base_dir: Path,
    model,
    subset_name: str,
    closest_train: bool,
    video_fps: int,
    video_output: Path | None,
    video_codec: str,
    log_message: str,
) -> None:
    n_out = len(dataset)
    if n_out == 0:
        raise Framework.InferenceError(f'{subset_name} produced no views.')
    renderer.render_subset(
        output_directory=base_dir / 'inference',
        dataset=dataset,
        calculate_metrics=False,
        visualize_errors=False,
        verbose=True,
        image_extension='png',
        save_gt=False,
        closest_train=closest_train,
    )
    frames_root = base_dir / 'inference' / f'{subset_name}_{model.num_iterations_trained}'
    rgb_dir = frames_root / 'rgb'
    out_mp4 = video_output
    if out_mp4 is None:
        out_mp4 = base_dir / 'inference' / f'{subset_name}_{model.num_iterations_trained}_{video_fps}fps.mp4'
    _encode_rgb_pngs_to_mp4(
        rgb_dir=rgb_dir,
        output_mp4=out_mp4,
        fps=video_fps,
        num_frames=n_out,
        video_codec=video_codec,
    )
    Logger.log_info(f'{log_message}{out_mp4}')


def main(
    *,
    base_dir: Path,
    checkpoint_name: str,
    subsets: list[str] | None,
    calculate_metrics: bool,
    closest_train: bool,
    visualize_errors: bool,
    benchmark: bool,
    interpolate_video: bool,
    interpolate_steps: int,
    interpolate_reference_split: str,
    orbit_video: bool,
    orbit_steps: int,
    orbit_radius_scale: float,
    orbit_reference_split: str,
    recorded_trajectory: Path | None,
    recorded_steps: int,
    video_fps: int,
    video_output: Path | None,
    video_codec: str,
) -> None:
    n_video_modes = sum([interpolate_video, orbit_video, recorded_trajectory is not None])
    if n_video_modes > 1:
        raise Framework.InferenceError('Choose at most one of --interpolate-video, --orbit-video, and --recorded-trajectory.')

    # setup framework
    Framework.setup(config_path=str(base_dir / 'training_config.yaml'), require_custom_config=True)
    # load dataset, model, and renderer
    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH
    )
    model = MI.get_model(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=str(base_dir / 'checkpoints' / checkpoint_name),
    ).eval()
    renderer = MI.get_renderer(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        model=model
    )
    # List-order c2w interpolation (straight segments between GT poses)
    if interpolate_video:
        if interpolate_reference_split not in dataset.data:
            raise Framework.InferenceError(
                f'interpolated_path: unknown reference split {interpolate_reference_split!r}. '
                f'Known keys: {list(dataset.data.keys())}.'
            )
        if not dataset.data[interpolate_reference_split]:
            raise Framework.InferenceError(
                f'interpolated_path: reference split {interpolate_reference_split!r} has no views.'
            )
        trajectory = CameraTrajectory.get('interpolated_path')(subdivisions_per_segment=interpolate_steps)
        trajectory.add_to_dataset(dataset, reference_set=interpolate_reference_split)
        dataset.set_mode(trajectory.name)
        _render_trajectory_to_mp4(
            renderer=renderer,
            dataset=dataset,
            base_dir=base_dir,
            model=model,
            subset_name=trajectory.name,
            closest_train=closest_train,
            video_fps=video_fps,
            video_output=video_output,
            video_codec=video_codec,
            log_message='Wrote pose-to-pose interpolation video to ',
        )

    # Horizontal orbit; angular order follows pose list (GUI next-pose order)
    if orbit_video:
        if orbit_reference_split not in dataset.data:
            raise Framework.InferenceError(
                f'pose_ordered_orbit: unknown reference split {orbit_reference_split!r}. '
                f'Known keys: {list(dataset.data.keys())}.'
            )
        if not dataset.data[orbit_reference_split]:
            raise Framework.InferenceError(
                f'pose_ordered_orbit: reference split {orbit_reference_split!r} has no views.'
            )
        trajectory = CameraTrajectory.get('pose_ordered_orbit')(
            subdivisions_per_segment=orbit_steps,
            radius_scale=orbit_radius_scale,
        )
        trajectory.add_to_dataset(dataset, reference_set=orbit_reference_split)
        dataset.set_mode(trajectory.name)
        _render_trajectory_to_mp4(
            renderer=renderer,
            dataset=dataset,
            base_dir=base_dir,
            model=model,
            subset_name=trajectory.name,
            closest_train=closest_train,
            video_fps=video_fps,
            video_output=video_output,
            video_codec=video_codec,
            log_message='Wrote pose-ordered orbit video to ',
        )

    if recorded_trajectory is not None:
        trajectory = CameraTrajectory.get('recorded_path')(
            trajectory_path=recorded_trajectory,
            subdivisions_per_segment=recorded_steps,
        )
        trajectory.add_to_dataset(dataset, reference_set=None)
        dataset.set_mode(trajectory.name)
        _render_trajectory_to_mp4(
            renderer=renderer,
            dataset=dataset,
            base_dir=base_dir,
            model=model,
            subset_name=trajectory.name,
            closest_train=closest_train,
            video_fps=video_fps,
            video_output=video_output,
            video_codec=video_codec,
            log_message='Wrote recorded GUI trajectory video to ',
        )

    # render subsets
    if subsets:
        if 'all' in subsets:
            subsets = dataset.subsets + CameraTrajectory.list_options()
        subsets = list(set(subsets))
        subsets.sort()
        if interpolate_video and 'interpolated_path' in subsets:
            subsets = [s for s in subsets if s != 'interpolated_path']
        if orbit_video and 'pose_ordered_orbit' in subsets:
            subsets = [s for s in subsets if s != 'pose_ordered_orbit']
        for subset in subsets:
            if subset not in dataset.subsets:
                if subset not in CameraTrajectory.list_options():
                    Logger.log_warning(f'Skipping unknown subset or camera trajectory: {subset}.')
                    continue
                ref_set = _trajectory_reference_set(subset, interpolate_reference_split, orbit_reference_split)
                if subset == 'interpolated_path':
                    if interpolate_reference_split not in dataset.data or not dataset.data[interpolate_reference_split]:
                        raise Framework.InferenceError(
                            f'interpolated_path: invalid or empty reference split {interpolate_reference_split!r}.'
                        )
                if subset == 'pose_ordered_orbit':
                    if orbit_reference_split not in dataset.data or not dataset.data[orbit_reference_split]:
                        raise Framework.InferenceError(
                            f'pose_ordered_orbit: invalid or empty reference split {orbit_reference_split!r}.'
                        )
                if subset == 'recorded_path':
                    if recorded_trajectory is None:
                        raise Framework.InferenceError('recorded_path requires --recorded-trajectory path.json.')
                    trajectory = CameraTrajectory.get(subset)(
                        trajectory_path=recorded_trajectory,
                        subdivisions_per_segment=recorded_steps,
                    )
                else:
                    trajectory = CameraTrajectory.get(subset)()
                trajectory.add_to_dataset(dataset, reference_set=ref_set)
            dataset.set_mode(subset)
            renderer.render_subset(
                output_directory=base_dir / 'inference',
                dataset=dataset,
                calculate_metrics=calculate_metrics,
                visualize_errors=visualize_errors,
                verbose=True,
                image_extension='png',
                save_gt=False,
                closest_train=closest_train,
            )
    # performance benchmark
    if benchmark:
        NUM_ITERATIONS = 100  # number of times the test set is rendered to calculate the online FPS
        Logger.log_info('Benchmarking online FPS (this might take a while)...')
        # if no test images are available, use the training set
        if len(dataset.test()) == 0:
            if len(dataset.train()) == 0:
                raise Framework.InferenceError('No images found for benchmarking.')
        # warmup
        for view in Logger.log_progress(dataset, leave=False, desc='Warming Up'):
            renderer.render_image(view, benchmark=True)
        # render
        num_test_images = len(dataset)
        torch.cuda.synchronize()
        start_time = perf_counter()
        for _ in Logger.log_progress(range(NUM_ITERATIONS), leave=False, desc='Benchmarking Performance'):
            for view in dataset:
                renderer.render_image(view, benchmark=True)
        torch.cuda.synchronize()
        end_time = perf_counter()
        # write output
        benchmark_output_path = base_dir / f'performance_{model.num_iterations_trained}.txt'
        total_time = end_time - start_time
        total_time_ms = total_time * 1000
        total_num_images = NUM_ITERATIONS * num_test_images
        avg_fps = total_num_images / total_time
        avg_ms_per_image = total_time_ms / total_num_images
        with open(str(benchmark_output_path), 'w') as f:
            f.write(f'Number of test set renders: {NUM_ITERATIONS}\n')
            f.write(f'Number of test set images: {num_test_images}\n')
            f.write(f'Test set image size: {view.camera.width}x{view.camera.height}\n')
            f.write(f'Total rendering time: {total_time_ms:.2f} ms\n')
            f.write(f'Average rendering time per image: {avg_ms_per_image:.2f} ms\n')
            f.write(f'Average FPS: {avg_fps:.2f}\n')
        Logger.log_info(f'Average FPS: {avg_fps:.2f} ({avg_ms_per_image:.2f} ms)')
        Logger.log_info(f'Performance benchmark results written to {benchmark_output_path}.')
    Logger.log_info('Done.')


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='inference.py',
        description=(
            'Renders outputs from a pretrained model. '
            '--orbit-video: horizontal orbit around mean camera position, angles follow pose list order. '
            '--interpolate-video: straight c2w segments between consecutive poses (GUI order). '
            'Both need ffmpeg on PATH for MP4.'
        )
    )
    parser.add_argument(
        '-d', '--dir', action='store', dest='base_dir', default=None,
        metavar='path/to/output/directory', required=True,
        help='A directory containing the outputs of a completed training.'
    )
    parser.add_argument(
        '-s', '--subsets', action='store', dest='subsets', default=None,
        metavar='subsets', required=False, nargs='*', type=str,
        help='Dataset subsets to render. If "all", all available subsets are rendered.'
    )
    parser.add_argument(
        '-m', '--metrics', action='store_true', dest='calculate_metrics',
        help='Calculates standard metrics for the rendered images, if ground truth is available.'
    )
    parser.add_argument(
        '-b', '--benchmark', action='store_true', dest='benchmark',
        help='Calculates the online FPS by repeatedly rendering the test set without output visualization/saving.'
    )
    parser.add_argument(
        '--closest_train', action='store_true', dest='closest_train',
        help='Renders the closest ground truth training image to the requested camera pose.'
    )
    parser.add_argument(
        '--visualize_errors', action='store_true', dest='visualize_errors',
        help='Visualizes the errors between the rendered and ground truth images, if available.'
    )
    parser.add_argument(
        '--checkpoint', action='store', dest='checkpoint_name', default='final.pt',
        metavar='checkpoint_name', required=False,
        help='The name of the checkpoint file to use for inference.'
    )
    parser.add_argument(
        '--interpolate-video', action='store_true', dest='interpolate_video',
        help=(
            'Render interpolated_path: straight motion between consecutive poses in list order '
            '(GUI pose prev/next order), then mux MP4.'
        )
    )
    parser.add_argument(
        '--interpolate-reference-split', action='store', dest='interpolate_reference_split', default='train',
        metavar='split',
        help='Reference split for --interpolate-video / -s interpolated_path (default: train).'
    )
    parser.add_argument(
        '--interpolate-steps', action='store', dest='interpolate_steps', type=int, default=10,
        metavar='N', required=False,
        help='Subdivisions per pose pair for --interpolate-video (default: 10).'
    )
    parser.add_argument(
        '--orbit-video', action='store_true', dest='orbit_video',
        help=(
            'Render pose_ordered_orbit: cameras on a horizontal circle around the mean camera position, '
            'looking at that center; azimuth advances in dataset list order (unwrap along indices), '
            'height lerps between consecutive poses. Then mux MP4.'
        )
    )
    parser.add_argument(
        '--orbit-reference-split', action='store', dest='orbit_reference_split', default='train',
        metavar='split',
        help='Reference split for --orbit-video / -s pose_ordered_orbit (default: train).'
    )
    parser.add_argument(
        '--orbit-steps', action='store', dest='orbit_steps', type=int, default=10,
        metavar='N', required=False,
        help='Subdivisions per pose pair for --orbit-video (default: 10).'
    )
    parser.add_argument(
        '--orbit-radius-scale', action='store', dest='orbit_radius_scale', type=float, default=1.0,
        metavar='s', required=False,
        help='Scales mean horizontal camera radius for --orbit-video (default: 1).'
    )
    parser.add_argument(
        '--video-fps', action='store', dest='video_fps', type=int, default=30,
        metavar='fps', required=False,
        help='Frame rate for trajectory MP4s (default: 30).'
    )
    parser.add_argument(
        '--recorded-trajectory', action='store', dest='recorded_trajectory', default=None,
        metavar='path.json', required=False,
        help='Render a GUI-recorded trajectory JSON to MP4.'
    )
    parser.add_argument(
        '--recorded-steps', action='store', dest='recorded_steps', type=int, default=1,
        metavar='N', required=False,
        help='Subdivisions per recorded keyframe pair (default: 1, replay exact recording samples).'
    )
    parser.add_argument(
        '--video-output', action='store', dest='video_output', default=None,
        metavar='path.mp4', required=False,
        help='Output MP4 path (default: inference/<trajectory>_<iter>_<fps>fps.mp4).'
    )
    parser.add_argument(
        '--video-codec', action='store', dest='video_codec', default='libx264',
        metavar='name', required=False,
        help='ffmpeg video encoder (default: libx264).'
    )
    args, _ = parser.parse_known_args()
    Logger.set_mode(Logger.MODE_VERBOSE)
    main(
        base_dir=Path(args.base_dir),
        checkpoint_name=args.checkpoint_name,
        subsets=args.subsets,
        calculate_metrics=args.calculate_metrics,
        closest_train=args.closest_train,
        visualize_errors=args.visualize_errors,
        benchmark=args.benchmark,
        interpolate_video=args.interpolate_video,
        interpolate_steps=args.interpolate_steps,
        interpolate_reference_split=args.interpolate_reference_split,
        orbit_video=args.orbit_video,
        orbit_steps=args.orbit_steps,
        orbit_radius_scale=args.orbit_radius_scale,
        orbit_reference_split=args.orbit_reference_split,
        recorded_trajectory=Path(args.recorded_trajectory) if args.recorded_trajectory else None,
        recorded_steps=args.recorded_steps,
        video_fps=args.video_fps,
        video_output=Path(args.video_output) if args.video_output else None,
        video_codec=args.video_codec,
    )
