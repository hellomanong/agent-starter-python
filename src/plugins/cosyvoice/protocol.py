from __future__ import annotations


def is_non_audio_content_type(content_type: str) -> bool:
    """Allow only raw/audio responses from the official CosyVoice server."""
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    return not (ct.startswith("audio/") or ct in ("application/octet-stream", ""))
