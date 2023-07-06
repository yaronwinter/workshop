import torch
import pandas as pd
import numpy as np
from scipy.io import wavfile
import math

ADDITIVE_INTENSITY = 96

def generate_train_set(raw_set_file: str):
    df = pd.read_csv(raw_set_file)

def compute_intensity(start_time: float, end_time: float, fs: float, audio_signal: np.ndarray) -> float:
    sub_signal = audio_signal[int(start_time*fs):int(end_time*fs)]
    sub_signal = sub_signal / np.max(np.abs(sub_signal))
    sub_signal = sub_signal**2
    rms = 10 * math.log10(rms)
    return rms + ADDITIVE_INTENSITY

def compute_pitch(start_time: float, end_time: float, fs: float, audio_signal: np.ndarray) -> float:
    sub_signal = audio_signal[int(start_time*fs):int(end_time*fs)]
