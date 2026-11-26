"""Train and render oriented two-axis Gaussian disks."""
from .Model import FasterGS2DModel
from .Renderer import FasterGS2DRenderer
from .Trainer import FasterGS2DTrainer

MODEL = FasterGS2DModel
RENDERER = FasterGS2DRenderer
TRAINING_INSTANCE = FasterGS2DTrainer
