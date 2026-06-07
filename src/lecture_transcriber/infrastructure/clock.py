"""System clock adapter returning the wall-clock as an aware ``datetime``."""

from __future__ import annotations

from datetime import UTC, datetime

from lecture_transcriber.domain.ports import Clock


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


__all__ = ["SystemClock"]
