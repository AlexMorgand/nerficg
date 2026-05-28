"""Horizontal orbit around scene center; angular sweep follows dataset pose list order."""

from copy import deepcopy

import numpy as np

import Framework
from Cameras.Base import BaseCamera
from Cameras.utils import look_at
from Datasets.utils import View
from Visual.Trajectories.utils import CameraTrajectory


def _unwrap_theta_along_index(thetas: np.ndarray) -> np.ndarray:
    """Cumulative angles: each step uses wrapped delta in (-pi, pi] so the path follows pose order."""
    if thetas.size == 0:
        return thetas
    out = np.empty_like(thetas)
    out[0] = thetas[0]
    for i in range(1, thetas.size):
        d = thetas[i] - thetas[i - 1]
        d = (d + np.pi) % (2.0 * np.pi) - np.pi
        out[i] = out[i - 1] + d
    return out


def _dataset_up_hint_for_look_at(reference_views: list[View]) -> np.ndarray:
    """
    Match ``spiral_path`` / ``get_average_pose``: ``look_at``'s third argument is the negative
    of the summed camera ``down`` axes (c2w column 1), not raw world +Y — otherwise images read flipped.
    """
    down_sum = np.stack([v.c2w_numpy[:3, 1] for v in reference_views], axis=0).sum(axis=0).astype(np.float64)
    if np.linalg.norm(down_sum) < 1e-9:
        d0 = reference_views[0].c2w_numpy[:3, 1].astype(np.float64)
        if np.linalg.norm(d0) < 1e-9:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return -d0
    return -down_sum


class pose_ordered_orbit(CameraTrajectory):
    """
    Cameras sit on a horizontal circle (xz) around the mean camera position, at mean radial distance,
    looking at that center. The polar angle advances along **dataset list order**: each GT pose
    contributes an azimuth; angles are unwrapped along the index so the orbit follows capture order
    (like GUI next pose) rather than sorting by angle.     Camera height (y) is linearly interpolated
    between consecutive poses; intrinsics follow the nearer keyframe (same idea as ``interpolated_path``).
    Orientation uses the same ``look_at`` up convention as ``spiral_path`` (negated sum of GT ``c2w`` column 1).
    """

    def __init__(self, subdivisions_per_segment: int = 10, radius_scale: float = 1.0, center: np.ndarray | None = None) -> None:
        super().__init__()
        if subdivisions_per_segment < 1:
            raise Framework.VisualizationError('subdivisions_per_segment must be >= 1')
        if radius_scale <= 0:
            raise Framework.VisualizationError('radius_scale must be > 0')
        self.subdivisions_per_segment = subdivisions_per_segment
        self.radius_scale = radius_scale
        self._center = np.asarray(center, dtype=np.float64).reshape(3) if center is not None else None

    def _generate(self, _: BaseCamera, reference_views: list[View]) -> list[View]:
        n = len(reference_views)
        if n == 0:
            return []
        positions = np.stack([v.position_numpy for v in reference_views], axis=0).astype(np.float64)

        if self._center is not None:
            center = self._center.copy()
        else:
            center = positions.mean(axis=0)

        up_for_look_at = _dataset_up_hint_for_look_at(reference_views)

        if n == 1:
            dx, dz = positions[0, 0] - center[0], positions[0, 2] - center[2]
            phi = float(np.arctan2(dx, dz))
            rh = float(np.hypot(dx, dz))
            R = max(rh * self.radius_scale, 1e-4)
            eye = np.array([
                center[0] + R * np.sin(phi),
                positions[0, 1],
                center[2] + R * np.cos(phi),
            ], dtype=np.float64)
            forward = center - eye
            fn = np.linalg.norm(forward)
            if fn < 1e-9:
                eye = center + np.array([0.0, 0.0, R], dtype=np.float64)
                forward = center - eye
                fn = np.linalg.norm(forward)
            ref = reference_views[0]
            return [View(
                camera=deepcopy(ref.camera),
                camera_index=ref.camera_index,
                frame_idx=0,
                global_frame_idx=0,
                c2w=look_at(eye, center, up_for_look_at).astype(np.float64),
                timestamp=0.0,
                exif=deepcopy(ref.exif),
            )]

        thetas = np.arctan2(positions[:, 0] - center[0], positions[:, 2] - center[2])
        horiz_r = np.hypot(positions[:, 0] - center[0], positions[:, 2] - center[2])
        R = float(np.mean(horiz_r)) * self.radius_scale
        R = max(R, float(np.percentile(horiz_r, 50)) * self.radius_scale * 0.25, 1e-4)

        phi_key = _unwrap_theta_along_index(thetas)
        ys = positions[:, 1]

        M = self.subdivisions_per_segment
        views_out: list[View] = []

        for seg in range(n - 1):
            for j in range(M + 1):
                if seg > 0 and j == 0:
                    continue
                t = j / float(M)
                phi = (1.0 - t) * phi_key[seg] + t * phi_key[seg + 1]
                y_eye = (1.0 - t) * ys[seg] + t * ys[seg + 1]
                eye = np.array([
                    center[0] + R * np.sin(phi),
                    y_eye,
                    center[2] + R * np.cos(phi),
                ], dtype=np.float64)
                forward = center - eye
                if np.linalg.norm(forward) < 1e-9:
                    continue
                ref = reference_views[seg] if t < 0.5 else reference_views[seg + 1]
                c2w = look_at(eye, center, up_for_look_at).astype(np.float64)
                views_out.append(View(
                    camera=deepcopy(ref.camera),
                    camera_index=ref.camera_index,
                    frame_idx=len(views_out),
                    global_frame_idx=len(views_out),
                    c2w=c2w,
                    timestamp=float(seg) + t,
                    exif=deepcopy(ref.exif),
                ))
        return views_out
