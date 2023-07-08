import torch
import pandas as pd
import numpy as np
from scipy.io import wavfile
import math
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import time

ADDITIVE_INTENSITY = 96
ID_FIELD = "ID"
START_TIME_FIELD = "xmin"
END_TIME_FIELD = "xmax"
LABEL_FIELD = "label"
PREV_DIST_FIELD = "prev_dist"
NEXT_DIST_FIELD = "next_dist"
DURATION_FIELD = "duration"
INTENSITY_FIELD = "intensity"
START_PITCH_FIELD = "start_pitch"
MIDDLE_PITCH_FIELD = "middle_pitch"
END_PITCH_FIELD = "end_pitch"

LOWEST_FREQUENCY = 60
HIGHEST_FREQUENCY = 300
MIN_SIGNAL_LENGTH = 0.2

def generate_train_set(raw_set_file: str, audio_file: str) -> pd.DataFrame:
    start_time = time.time()
    df = pd.read_csv(raw_set_file)
    fs, audio_signal = wavfile.read(audio_file)

    df[DURATION_FIELD] = df[END_TIME_FIELD] - df[START_TIME_FIELD]
    df[PREV_DIST_FIELD] = df[START_TIME_FIELD] - df[END_TIME_FIELD].shift().fillna(0)
    df[NEXT_DIST_FIELD] = df[START_TIME_FIELD].shift(periods=-1).fillna(df[END_TIME_FIELD].max()) - df[END_TIME_FIELD]
    df[INTENSITY_FIELD] = df.apply(lambda r: compute_intensity(r[START_TIME_FIELD], r[END_TIME_FIELD], fs, audio_signal), axis=1)

    audio_signal, fs = torchaudio.load(audio_file)
    pitch_df = df.apply(lambda r: compute_pitch(r[START_TIME_FIELD], r[END_TIME_FIELD], fs, audio_signal), axis=1)
    pitch_df = pd.DataFrame({"pitches": pitch_df.values})
    df[START_PITCH_FIELD] = pitch_df.pitches.apply(lambda r: r[START_PITCH_FIELD])
    df[MIDDLE_PITCH_FIELD] = pitch_df.pitches.apply(lambda r: r[MIDDLE_PITCH_FIELD])
    df[END_PITCH_FIELD] = pitch_df.pitches.apply(lambda r: r[END_PITCH_FIELD])

    print("generate train set run time: {:.2f}".format(time.time() - start_time))

    return df

def compute_intensity(start_time: float, end_time: float, fs: float, audio_signal: np.ndarray) -> float:
    sub_signal = audio_signal[int(start_time*fs):int(end_time*fs)]
    sub_signal = sub_signal / np.max(np.abs(sub_signal))
    sub_signal = sub_signal**2
    rms = 10 * math.log10(sub_signal.sum() / sub_signal.shape[0])
    return rms + ADDITIVE_INTENSITY

def compute_pitch(start_time: float, end_time: float, fs: int, audio_signal: torch.Tensor) -> dict:
    signal_length = end_time - start_time
    actual_end_time = (end_time if signal_length >= MIN_SIGNAL_LENGTH else (start_time + MIN_SIGNAL_LENGTH))
    pitch = F.detect_pitch_frequency(torch.unsqueeze(audio_signal[0,int(start_time*fs):int(actual_end_time*fs)], 0), fs, freq_low=LOWEST_FREQUENCY, freq_high=HIGHEST_FREQUENCY)
    num_items = pitch.shape[1]
    one_third = int(num_items/3 + 0.5)
    return {START_PITCH_FIELD:(pitch[0,:one_third].sum()/one_third).item(),
            MIDDLE_PITCH_FIELD:(pitch[0,one_third:2*one_third].sum()/one_third).item(),
            END_PITCH_FIELD:(pitch[0,2*one_third:].sum()/(num_items - 2*one_third)).item()}
