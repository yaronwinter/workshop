from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
import pandas as pd
from prosodic import utils as prosodic_utils
import torch
import torchaudio

BATCH_SIZE = "batch_size"
RANDOM_SAMPLING= "random"
SEQUENTIAL_SAMPLING = "sequential"

def df_to_dataloader(df: pd.DataFrame, config: dict, sampling_type: str) -> DataLoader:
    audio_signal, fs = torchaudio.load(config[prosodic_utils.AUDIO_FILE])
    max_length = fs * config[prosodic_utils.DNN_DURATION].tolist()

    ids = df[prosodic_utils.ID_FIELD].to_list()
    start_times = df[prosodic_utils.DNN_START_TIME].to_list()
    end_times = df[prosodic_utils.DNN_END_TIME].to_list()
    labels = df[prosodic_utils.Y_FIELD].to_list()

    signals = []
    lengths = []
    for start_time, end_time in zip(start_times, end_times):
        start_ind = int(start_time * fs)
        end_ind = int(end_time * fs)
        length = end_ind - start_ind
        signal = audio_signal[0, start_ind:end_ind].tolist()
        signal += [0] * (max_length - length)
        
        signals.append(signal)
        lengths.append(length)
        
    signals, labels, lengths, ids = tuple(torch.tensor(data) for data in [signals, labels, lengths, lengths, ids])
    data = TensorDataset(signals, labels, lengths, ids)
    
    if sampling_type == RANDOM_SAMPLING:
        sampler = RandomSampler(data)
    elif sampling_type == SEQUENTIAL_SAMPLING:
        sampler = SequentialSampler(data)
    else:
        print('Wrong Sampling Type: ' + sampling_type)
        return None
        
    dataloader = DataLoader(data, sampler=sampler, batch_size=config[BATCH_SIZE])
    return dataloader
