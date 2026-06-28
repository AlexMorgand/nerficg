"""Faster2DGS/Model.py

2D surfel Gaussians (``Gaussians2D``) with native diff-surfel rasterization (Phase C) or FasterGS bridge fallback.
"""

import Framework
from Methods.Faster2DGS.Gaussians2D import Gaussians2D
from Methods.FasterGS.Model import FasterGSModel
from Optim.ppisp import PPISPWrapper


@Framework.Configurable.configure(
    SH_DEGREE=3,
    PPISP=Framework.ConfigParameterList(
        USE=False,
        CONTROLLER_TRAINING_STEPS=5_000,
        CONTROLLER_DISTILLATION=True,
    ),
)
class Faster2DGSModel(FasterGSModel):
    """Faster2DGS model with ``(N, 2)`` surfel scales."""

    def build(self) -> 'Faster2DGSModel':
        pretrained = self.num_iterations_trained > 0
        self.gaussians = Gaussians2D(self.SH_DEGREE, pretrained)
        if self.PPISP.USE:
            self.ppisp = PPISPWrapper(self.PPISP)
        return self

    def get_ply_dict(self) -> dict:
        data = super().get_ply_dict()
        if not data:
            return data
        comments = list(data.get('comments', []))
        comments.append('Note: Faster2DGS Gaussians2D — 2D log-scales (scale_0, scale_1); surfel bridge rasterizer.')
        data['comments'] = comments
        return data
