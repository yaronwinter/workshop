import torch
import torchaudio
import torchaudio.transforms as T

N_FFT = "n_fft"
HOP_LENGTH = "hop_length"
WINDOW_LENGTH = "win_length"

def get_spectrogram(audio_file: str, start_time: float, end_time: float, spec_params: dict) -> torch.Tensor:
    spec = T.Spectrogram(
        n_fft=spec_params[N_FFT],
        win_length=spec_params[WINDOW_LENGTH],
        hop_length=spec_params[HOP_LENGTH],
        center=True,
        pad_mode="reflect",
        power=2.0
    )

    audio_signal, fs = torchaudio.load(audio_file)
    start_ind = int(start_time*fs)
    end_ind = int(end_time*fs)
    active_signal = torch.unsqueeze(audio_signal[0,start_ind:end_ind], 0)
    return spec(active_signal)
