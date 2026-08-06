"""Media probing using PyAV.

Returns a :class:`MediaProbeResult` that records whether the file has a
decodable audio stream and basic audio metadata. Files without a usable audio
stream are rejected with :class:`MediaProbeFailed`.
"""

from __future__ import annotations

from pathlib import Path

import av

from lecture_transcriber.domain.errors import MediaProbeFailed
from lecture_transcriber.domain.ports import MediaProbe, MediaProbeResult


def _duration_seconds(container: object, stream: object) -> float:
    container_duration = getattr(container, "duration", None)
    if container_duration is not None:
        duration = float(container_duration) / 1_000_000.0
    else:
        stream_duration = getattr(stream, "duration", None)
        time_base = getattr(stream, "time_base", None)
        duration = (
            float(stream_duration) * float(time_base)
            if stream_duration is not None and time_base is not None
            else 0.0
        )
    if duration <= 0:
        raise MediaProbeFailed("media has no positive decodable duration")
    return duration


class PyAVMediaProbe(MediaProbe):
    def probe(self, path: Path) -> MediaProbeResult:
        try:
            with av.open(str(path)) as container:
                audio_streams = [s for s in container.streams if s.type == "audio"]
                if not audio_streams:
                    raise MediaProbeFailed(f"file {path.name} has no decodable audio stream")
                stream = audio_streams[0]
                audio_codec = stream.codec.name or "unknown"
                ctx = stream.codec_context
                # PyAV's stubs don't expose these; access via getattr to stay
                # mypy-clean without depending on private attributes.
                sample_rate = getattr(ctx, "sample_rate", None)
                channels = getattr(ctx, "channels", None)
                if sample_rate is not None:
                    sample_rate = int(sample_rate) or None
                if channels is not None:
                    channels = int(channels) or None

                duration = _duration_seconds(container, stream)

                media_type = (
                    "video" if any(s.type == "video" for s in container.streams) else "audio"
                )

                return MediaProbeResult(
                    media_type=media_type,  # type: ignore[arg-type]
                    duration_seconds=duration,
                    audio_codec=audio_codec,
                    audio_sample_rate=sample_rate,
                    audio_channels=channels,
                )
        except MediaProbeFailed:
            raise
        except Exception as exc:
            raise MediaProbeFailed(f"failed to probe {path.name}: {exc}") from exc
