from __future__ import annotations

from array import array

from livekit import rtc

from .config import INPUT_SAMPLE_RATE


def downmix_to_mono(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    if frame.num_channels == 1:
        return frame

    samples = array("h")
    samples.frombytes(bytes(frame.data))

    if len(samples) % frame.num_channels != 0:
        raise ValueError("audio data length is not divisible by channel count")

    mono = array("h")
    for idx in range(0, len(samples), frame.num_channels):
        mixed = int(sum(samples[idx : idx + frame.num_channels]) / frame.num_channels)
        mono.append(mixed)

    return rtc.AudioFrame(
        data=mono.tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=1,
        samples_per_channel=len(mono),
    )


def resample_to_input_rate(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    if frame.sample_rate == INPUT_SAMPLE_RATE:
        return frame

    resampler = rtc.AudioResampler(
        input_rate=frame.sample_rate,
        output_rate=INPUT_SAMPLE_RATE,
        num_channels=frame.num_channels,
        quality=rtc.AudioResamplerQuality.HIGH,
    )
    resampled = resampler.push(frame) + resampler.flush()
    if not resampled:
        return rtc.AudioFrame(
            data=b"",
            sample_rate=INPUT_SAMPLE_RATE,
            num_channels=frame.num_channels,
            samples_per_channel=0,
        )

    return rtc.combine_audio_frames(resampled)


def normalize_audio_frame(frame: rtc.AudioFrame) -> rtc.AudioFrame:
    return resample_to_input_rate(downmix_to_mono(frame))
