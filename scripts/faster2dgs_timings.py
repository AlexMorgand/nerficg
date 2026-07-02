"""Parse and report Faster2DGS training / step timings for eval and parity runs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CallbackTiming:
    name: str
    total_sec: float
    ms_per_iter: float
    iterations: int


@dataclass
class TrainingTimings:
    total_sec: float | None = None
    callbacks: list[CallbackTiming] = field(default_factory=list)

    @property
    def training_iteration(self) -> CallbackTiming | None:
        for cb in self.callbacks:
            if cb.name == 'training_iteration':
                return cb
        return None

    @property
    def ms_per_iter(self) -> float | None:
        ti = self.training_iteration
        return ti.ms_per_iter if ti else None

    @property
    def train_sec(self) -> float | None:
        ti = self.training_iteration
        return ti.total_sec if ti else None


@dataclass
class StepProfile:
    iteration: int
    splats: int
    total_ms: float
    ms_per_iter: float
    steps_per_s: float
    aux_mode: int | None = None
    components: dict[str, float] = field(default_factory=dict)


_DURATION = re.compile(
    r'^(?P<name>.+):\s*\n'
    r'\tTotal execution time: (?P<total>[\d:]+(?::\d\d)?)\s*\n'
    r'\tTime per iteration \[ms\]: (?P<ms>[\d.]+)\s*\n'
    r'\tNumber of iterations: (?P<n>\d+)',
    re.MULTILINE,
)
_TOTAL = re.compile(r'^Time:(?P<sec>[\d.]+)\s*$', re.MULTILINE)


def _parse_duration(text: str) -> float:
    parts = text.strip().split(':')
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    hours, minutes, seconds = (int(p) for p in parts)
    return hours * 3600 + minutes * 60 + seconds


def parse_timings_txt(path: Path) -> TrainingTimings:
    if not path.is_file():
        return TrainingTimings()
    text = path.read_text()
    callbacks = [
        CallbackTiming(
            name=m.group('name'),
            total_sec=_parse_duration(m.group('total')),
            ms_per_iter=float(m.group('ms')),
            iterations=int(m.group('n')),
        )
        for m in _DURATION.finditer(text)
    ]
    total = None
    if m := _TOTAL.search(text):
        total = float(m.group('sec'))
    return TrainingTimings(total_sec=total, callbacks=callbacks)


def write_timing_summary(run_dir: Path, summary: dict) -> Path:
    run_dir = run_dir.resolve()
    out = run_dir / 'timing_summary.json'
    out.write_text(json.dumps(summary, indent=2) + '\n')
    return out


def format_train_timing(t: TrainingTimings) -> str | None:
    ti = t.training_iteration
    if ti is None:
        return None
    its = 1000.0 / ti.ms_per_iter if ti.ms_per_iter > 0 else 0.0
    return f'train {ti.total_sec:.0f}s ({ti.ms_per_iter:.1f} ms/iter, {its:.1f} it/s)'


def format_step_profile(p: StepProfile) -> str:
    aux = {0: 'photometric', 1: 'distortion', 2: 'full'}.get(p.aux_mode or -1, str(p.aux_mode))
    return (
        f'step@{p.iteration:,}: {p.total_ms:.1f} ms ({p.steps_per_s:.1f} it/s, '
        f'{p.splats:,} splats, aux={aux})'
    )


def profile_training_steps(
    trainer,
    dataset,
    iterations: list[int],
    *,
    repeats: int = 15,
    warmup: int = 5,
) -> list[StepProfile]:
    """CUDA step breakdown using the post-train model state."""
    from Logging import Logger
    from faster2dgs_profile_step import profile_step

    profiles: list[StepProfile] = []
    end = int(trainer.model.num_iterations_trained)
    if end <= 0 or trainer.model.gaussians.means.shape[0] <= 0:
        return profiles

    for iteration in iterations:
        if iteration < end - 50:
            Logger.log_warning(
                f'skip step profile @ {iteration}: training finished at {end} '
                f'(splats are end-state). Use faster2dgs_profile_step.py --advance-to {iteration}.'
            )
            continue
        it = min(max(iteration, 0), end - 1)
        aux_mode = trainer._aux_mode(it)
        timings = profile_step(trainer, dataset, it, repeats=repeats, warmup=warmup)
        total_ms = timings.total_ms()
        profiles.append(
            StepProfile(
                iteration=it,
                splats=trainer.model.gaussians.means.shape[0],
                total_ms=total_ms,
                ms_per_iter=total_ms,
                steps_per_s=1000.0 / total_ms if total_ms > 0 else 0.0,
                aux_mode=aux_mode,
                components=dict(zip(timings.labels, timings.ms)),
            )
        )
    return profiles


def collect_run_timings(
    run_dir: Path,
    *,
    profile_steps: list[StepProfile] | None = None,
) -> dict:
    train = parse_timings_txt(run_dir / 'timings.txt')
    summary: dict = {
        'run_dir': str(run_dir.resolve()),
        'training': {
            'total_sec': train.total_sec,
            'training_iteration_sec': train.train_sec,
            'ms_per_iter': train.ms_per_iter,
            'callbacks': [asdict(cb) for cb in train.callbacks],
        },
    }
    if profile_steps:
        summary['step_profiles'] = [asdict(p) for p in profile_steps]
    write_timing_summary(run_dir, summary)
    return summary
