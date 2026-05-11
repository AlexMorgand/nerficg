"""Faster2DGS/Model.py

``Faster2DGS`` is **not** a separate surfel rasterizer in this repo. It subclasses
:class:`~Methods.FasterGS.Model.FasterGSModel` and therefore uses the **same**
full 3D Gaussian primitives (three unconstrained scales + quaternion) and the
same FasterGS CUDA forward / backward as ``FasterGS``.

What differs is training/inference plumbing:

- Training calls ``diff_rasterize_with_aux``, which asks the FasterGS kernel to
  accumulate extra per-pixel buffers (opacity, depth, distortion proxy, and a
  direction derived from the Gaussian's rotation matrix).

- The auxiliary **normal** direction written in CUDA is the **third column of the
  rotation matrix** ``R`` (local ``+Z`` expressed in world space). That matches a
  **flat disk in the local XY plane** of each primitive **only if** the third
  log-scale (the axis multiplied onto that column in ``RSS``) is made thin.

Export viewers read the same PLY layout as FasterGS, so **splats are drawn as
3D ellipsoids** unless an external viewer applies special shading — nothing here
switches the viewer to 2D disks automatically.

To encourage pancake-like splats (closer to classical 2DGS geometry), use the
optional planar-scale regularization on ``raw_scales[:, 2]`` in the 2DGS loss.
"""

from Methods.FasterGS.Model import FasterGSModel


class Faster2DGSModel(FasterGSModel):
    """FasterGS model used with :class:`~Methods.Faster2DGS.Renderer.Faster2DGSRenderer` and aux buffers."""

    def get_ply_dict(self) -> dict:
        """Same PLY as FasterGS; comments record that the run was Faster2DGS."""
        data = super().get_ply_dict()
        if not data:
            return data
        comments = list(data.get('comments', []))
        comments.append(
            'Note: Faster2DGS uses FasterGS 3D Gaussians + aux raster buffers; not a separate 2D surfel kernel.'
        )
        data['comments'] = comments
        return data
