"""Interpolate cameras along dataset list order (same keyframe order as GUI pose prev/next)."""

from copy import deepcopy

import numpy as np
import torch

import Framework
from Cameras.utils import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from Datasets.utils import View
from Visual.Trajectories.utils import CameraTrajectory


def _slerp_quaternion(q0: torch.Tensor, q1: torch.Tensor, t: float, eps: float = 1e-7) -> torch.Tensor:
    """Spherical linear interpolation between two unit quaternions (w, x, y, z)."""
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = (q0 * q1).sum().clamp(-1.0, 1.0)
    if dot.item() < 0.0:
        q1 = -q1
        dot = -dot
    if dot.item() > 1.0 - eps:
        return torch.nn.functional.normalize(q0 + float(t) * (q1 - q0), dim=-1)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * float(t)
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    return torch.nn.functional.normalize(s0 * q0 + s1 * q1, dim=-1)


def _interpolate_c2w(c0: np.ndarray, c1: np.ndarray, t: float) -> np.ndarray:
    """Rigid blend: lerp translation, slerp rotation from upper 3x3 of c2w."""
    R0 = torch.from_numpy(c0[:3, :3]).float().unsqueeze(0)
    R1 = torch.from_numpy(c1[:3, :3]).float().unsqueeze(0)
    q0 = rotation_matrix_to_quaternion(R0)[0]
    q1 = rotation_matrix_to_quaternion(R1)[0]
    q = _slerp_quaternion(q0, q1, t)
    R = quaternion_to_rotation_matrix(q.unsqueeze(0), normalize=True).squeeze(0).numpy().astype(np.float64)
    tw = ((1.0 - t) * c0[:3, 3] + t * c1[:3, 3]).astype(np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = tw
    return out


class interpolated_path(CameraTrajectory):
    """
    Smooth motion between consecutive poses in ``reference_views`` list order (indices 0→1→2→…),
    the same ordering as the GUI left/right pose keys for that split.

    Each segment uses ``subdivisions_per_segment`` steps from pose k to k+1 (inclusive endpoints);
    shared segment vertices are not duplicated. Translation is linear; rotation is quaternion slerp.

    For each frame, intrinsics are taken from the nearer segment endpoint (matches discrete GUI jumps
    at each keyframe better than freezing camera 0 for the whole path).
    """

    def __init__(self, subdivisions_per_segment: int = 10) -> None:
        super().__init__()
        if subdivisions_per_segment < 1:
            raise Framework.VisualizationError('subdivisions_per_segment must be >= 1')
        self.subdivisions_per_segment = subdivisions_per_segment

    def _generate(self, _, reference_views: list[View]) -> list[View]:
        n = len(reference_views)
        if n == 0:
            return []
        if n == 1:
            v = reference_views[0].to_simple()
            return [v]

        views_out: list[View] = []
        M = self.subdivisions_per_segment

        for seg in range(n - 1):
            c0 = reference_views[seg].c2w_numpy
            c1 = reference_views[seg + 1].c2w_numpy
            for j in range(M + 1):
                if seg > 0 and j == 0:
                    continue
                t = j / float(M)
                c2w = _interpolate_c2w(c0, c1, t)
                ref_cam = reference_views[seg] if t < 0.5 else reference_views[seg + 1]
                v = View(
                    camera=deepcopy(ref_cam.camera),
                    camera_index=ref_cam.camera_index,
                    frame_idx=len(views_out),
                    global_frame_idx=len(views_out),
                    c2w=c2w,
                    timestamp=float(seg) + t,
                    exif=deepcopy(ref_cam.exif),
                )
                views_out.append(v)
        return views_out
