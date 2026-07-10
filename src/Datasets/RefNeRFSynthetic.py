"""
Datasets/RefNeRFSynthetic.py: Loader for Ref-NeRF / NeRO-style synthetic reflective
scenes (Shiny Blender: ball, car, coffee, helmet, teapot, toaster; Glossy Synthetic).

Same ``transforms_{train,test}.json`` layout as the NeRF synthetic dataset, but:
  - no per-view depth maps (the reflective datasets do not ship them),
  - alpha is loaded only when the images actually carry an alpha channel (e.g. ball
    is stored as RGB with a baked background),
  - an optional ``points3d.ply`` is used to initialize the Gaussians,
  - ``WHITE_BACKGROUND`` (default True) matches 3DGS-DR ``--white-background``:
    RGBA frames are composited onto ``BACKGROUND_COLOR`` at load time and alpha
    is dropped so training uses a fixed background instead of random compositing.
"""

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import fov_to_focal
from Datasets.Base import BaseDataset
from Datasets.utils import (
    View,
    BasicPointCloud,
    compute_scaled_image_size,
    read_image_size,
    ImageData,
    load_image_composited_on_background,
)
from Logging import Logger


def _image_channel_count(path: Path) -> int:
    with Image.open(path) as img:
        return len(img.getbands())


def _load_point_cloud(path: Path) -> BasicPointCloud | None:
    if not path.exists():
        return None
    try:
        plydata = PlyData.read(str(path))
        vertices = plydata['vertex']
        positions = np.column_stack((vertices['x'], vertices['y'], vertices['z'])).astype(np.float32)
        names = [p.name for p in vertices.properties]
        colors = None
        if all(c in names for c in ('red', 'green', 'blue')):
            colors = np.column_stack((vertices['red'], vertices['green'], vertices['blue'])).astype(np.float32) / 255.0
    except Exception as exc:
        Logger.log_warning(f'failed to read point cloud "{path}": {exc}')
        return None
    point_cloud = BasicPointCloud(torch.from_numpy(positions))
    if colors is not None:
        point_cloud.colors = torch.from_numpy(colors)
    return point_cloud


@Framework.Configurable.configure(
    PATH='dataset/ref_nerf/ball',
    BACKGROUND_COLOR=[1.0, 1.0, 1.0],
    WHITE_BACKGROUND=True,  # 3DGS-DR --white-background: bake RGBA onto BACKGROUND_COLOR at load
    NORMALIZE_CUBE=4.0 / 1.5,
    NEAR_PLANE=2.0,
    FAR_PLANE=6.0,
    POINT_CLOUD_FILE='points3d.ply',
)
class CustomDataset(BaseDataset):
    """Dataset class for Ref-NeRF / NeRO synthetic reflective scenes."""

    def load(self) -> tuple[list[PerspectiveCamera], dict[str, list[View]]]:
        """Loads the dataset into a dict containing lists of views for the subsets."""
        camera = None
        cam_transform = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL to Colmap
        world_transform = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])  # Blender to Colmap
        self.bounding_box = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32, device='cpu')
        data: dict[str, list[View]] = {subset: [] for subset in self.subsets}
        global_frame_idx = 0
        white_background = bool(getattr(self, 'WHITE_BACKGROUND', False))
        if white_background:
            Logger.log_info(
                f'white-background mode: compositing RGBA onto BACKGROUND_COLOR={self.BACKGROUND_COLOR} at load (3DGS-DR parity)'
            )
            rgb_load_fn = partial(load_image_composited_on_background, background=self.BACKGROUND_COLOR)
        for subset in self.subsets:
            metadata_filepath: Path = self.dataset_path / f'transforms_{subset}.json'
            if not metadata_filepath.exists():
                # some subsets (e.g. val) may be absent for these datasets
                continue
            with open(metadata_filepath, 'r') as f:
                metadata_file: dict[str, Any] = json.load(f)

            frames = metadata_file['frames']
            has_alpha = False
            if not white_background:
                first_path = self.dataset_path / f'{frames[0]["file_path"]}'
                if first_path.suffix == '':
                    first_path = first_path.with_suffix(".png")
                has_alpha = _image_channel_count(first_path) >= 4

            for frame_idx, frame in Logger.log_progress(enumerate(frames), desc=subset, leave=False, total=len(frames)):
                rgba_path = self.dataset_path / f'{frame["file_path"]}'
                if rgba_path.suffix == '':
                    rgba_path = rgba_path.with_suffix(".png")
                width, height = compute_scaled_image_size(read_image_size(rgba_path), self.IMAGE_SCALE_FACTOR)
                focal = fov_to_focal(float(metadata_file['camera_angle_x'])) * width
                if camera is None:
                    camera = PerspectiveCamera(
                        shared_settings=self._camera_settings, width=width, height=height, focal_x=focal, focal_y=focal,
                    )
                elif camera.focal_x != focal or camera.width != width or height != camera.height:
                    raise Framework.DatasetError('the RefNeRFSynthetic loader requires all views to have the same image size and focal length.')

                c2w = world_transform @ frame['transform_matrix'] @ cam_transform.T
                if white_background:
                    rgb = ImageData(
                        rgba_path,
                        n_channels=3,
                        scale_factor=self.IMAGE_SCALE_FACTOR,
                        load_fn=rgb_load_fn,
                    )
                    alpha = None
                else:
                    rgb = ImageData(rgba_path, n_channels=3, scale_factor=self.IMAGE_SCALE_FACTOR)
                    alpha = ImageData(rgba_path, n_channels=1, channel_offset=3, scale_factor=self.IMAGE_SCALE_FACTOR) if has_alpha else None
                data[subset].append(View(
                    camera=camera,
                    camera_index=0,
                    frame_idx=frame_idx,
                    global_frame_idx=global_frame_idx,
                    c2w=c2w,
                    rgb=rgb,
                    alpha=alpha,
                ))
                global_frame_idx += 1

        # optional Gaussian initialization point cloud
        point_cloud = _load_point_cloud(self.dataset_path / self.POINT_CLOUD_FILE)
        if point_cloud is not None:
            self.point_cloud = point_cloud

        return [camera], data
