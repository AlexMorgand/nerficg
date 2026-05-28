"""Replay GUI-recorded camera trajectories."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import SharedCameraSettings
from Datasets.utils import View
from Visual.Trajectories.InterpolatedPath import _interpolate_c2w
from Visual.Trajectories.utils import CameraTrajectory


class recorded_path(CameraTrajectory):
    """Camera trajectory loaded from a GUI recording JSON file."""

    def __init__(self, trajectory_path: str | Path, subdivisions_per_segment: int = 1) -> None:
        super().__init__()
        self.trajectory_path = Path(trajectory_path).expanduser()
        if subdivisions_per_segment < 1:
            raise Framework.VisualizationError('subdivisions_per_segment must be >= 1')
        self.subdivisions_per_segment = subdivisions_per_segment

    def _load_frames(self) -> list[dict]:
        if not self.trajectory_path.is_file():
            raise Framework.VisualizationError(f'recorded trajectory file not found: {self.trajectory_path}')
        with open(self.trajectory_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        frames = payload.get('frames')
        if not isinstance(frames, list) or not frames:
            raise Framework.VisualizationError(f'recorded trajectory has no frames: {self.trajectory_path}')
        return frames

    @staticmethod
    def _camera_from_payload(camera_payload: dict, fallback_camera) -> PerspectiveCamera:
        if not isinstance(fallback_camera, PerspectiveCamera):
            raise Framework.VisualizationError('recorded_path currently supports PerspectiveCamera only.')
        bg = camera_payload.get('background_color', fallback_camera.background_color.detach().cpu().tolist())
        shared_settings = SharedCameraSettings(
            background_color=torch.tensor(bg, dtype=torch.float32),
            near_plane=float(camera_payload.get('near_plane', fallback_camera.near_plane)),
            far_plane=float(camera_payload.get('far_plane', fallback_camera.far_plane)),
        )
        return PerspectiveCamera(
            shared_settings=shared_settings,
            width=int(camera_payload.get('width', fallback_camera.width)),
            height=int(camera_payload.get('height', fallback_camera.height)),
            focal_x=float(camera_payload.get('focal_x', fallback_camera.focal_x)),
            focal_y=float(camera_payload.get('focal_y', fallback_camera.focal_y)),
            center_x=float(camera_payload.get('center_x', fallback_camera.center_x)),
            center_y=float(camera_payload.get('center_y', fallback_camera.center_y)),
            distortion=deepcopy(fallback_camera.distortion),
        )

    def _expand_frames(self, frames: list[dict]) -> list[tuple[np.ndarray, dict, float]]:
        if len(frames) == 1 or self.subdivisions_per_segment == 1:
            return [
                (np.asarray(frame['c2w'], dtype=np.float64), frame.get('camera', {}), float(frame.get('timestamp', frame.get('time', idx))))
                for idx, frame in enumerate(frames)
            ]

        expanded: list[tuple[np.ndarray, dict, float]] = []
        M = self.subdivisions_per_segment
        for seg in range(len(frames) - 1):
            f0 = frames[seg]
            f1 = frames[seg + 1]
            c0 = np.asarray(f0['c2w'], dtype=np.float64)
            c1 = np.asarray(f1['c2w'], dtype=np.float64)
            for j in range(M + 1):
                if seg > 0 and j == 0:
                    continue
                t = j / float(M)
                camera_payload = f0.get('camera', {}) if t < 0.5 else f1.get('camera', {})
                ts0 = float(f0.get('timestamp', f0.get('time', seg)))
                ts1 = float(f1.get('timestamp', f1.get('time', seg + 1)))
                expanded.append((_interpolate_c2w(c0, c1, t), camera_payload, (1.0 - t) * ts0 + t * ts1))
        return expanded

    def _generate(self, default_camera, _) -> list[View]:
        frames = self._expand_frames(self._load_frames())
        views: list[View] = []
        for idx, (c2w, camera_payload, timestamp) in enumerate(frames):
            views.append(View(
                camera=self._camera_from_payload(camera_payload, default_camera),
                camera_index=0,
                frame_idx=idx,
                global_frame_idx=idx,
                c2w=c2w,
                timestamp=timestamp,
                exif={},
            ))
        return views
