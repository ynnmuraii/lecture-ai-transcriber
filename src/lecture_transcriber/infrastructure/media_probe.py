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


class PyAVMediaProbe(MediaProbe):
    def probe(self, path: Path) -> MediaProbeResult:
        try:
            with av.open(str(path)) as container:
                audio_streams = [
                    s for s in container.streams if s.type == "audio"
                ]
                if not audio_streams:
                    raise MediaProbeFailed(
                        f"file {path.name} has no decodable audio stream"
                    )
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

                if container.duration is not None:
                    duration = float(container.duration) / 1_000_000.0
                else:
                    duration = float(stream.duration or 0.0) or 0.0

                media_type = "video" if any(
                    s.type == "video" for s in container.streams
                ) else "audio"

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
