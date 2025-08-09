import io
import torch
import av


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2**15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2**31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def load_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    """Decode a WAV/MP3/etc. from bytes → (waveform[C,T] float32, sr)."""
    with av.open(io.BytesIO(data)) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream in container")
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels
        frames, length = [], 0

        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:  # sometimes PyAV flattens
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)
            length += buf.shape[1]

        if not frames:
            raise ValueError("No audio frames decoded")
        wav = torch.cat(frames, dim=1)
        wav = f32_pcm(wav)  # convert to −1…1 float
        return wav, sr
