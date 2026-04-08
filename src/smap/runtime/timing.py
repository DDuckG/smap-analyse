from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

StageProgressCallback = Callable[[str, Literal["start", "end"], float | None], None]


@dataclass(slots=True)
class StageTimingCollector:
    stage_seconds: dict[str, float] = field(default_factory=dict)
    progress_callback: StageProgressCallback | None = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.progress_callback is not None:
            self.progress_callback(name, "start", None)
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = round(perf_counter() - started, 4)
            self.stage_seconds[name] = elapsed
            if self.progress_callback is not None:
                self.progress_callback(name, "end", elapsed)

    @property
    def total_seconds(self) -> float:
        return round(sum(self.stage_seconds.values()), 4)
