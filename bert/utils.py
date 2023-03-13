import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from utils import config as params

ATTENTION_MASK = "attention_mask"
INPUT_IDS = "input_ids"
MAX_LENGTH = "max_length"
NUM_ADDITIONAL_TOKENS = 2
PYTORCH_CODE = "pt"

def df_to_dataloader(df: pd.DataFrame, tokenizer, config, sampling_type) -> TensorDataset:
    sorted_df = df.sort_values(by=config[params.LENGTH_COL], axis=0, ascending=False, ignore_index=True)
    labels = sorted_df[config[params.LABEL_COL]].values.tolist()
    texts = sorted_df[config[params.TEXT_COL]].values.tolist()
    max_len = sorted_df[config[params.LENGTH_COL]].max() + NUM_ADDITIONAL_TOKENS

    tokenized_texts = [tokenizer(text, padding='max_length', max_length=max_len, truncation=True, return_tensors=PYTORCH_CODE) for text in texts]
    indexed_texts = torch.stack([item[INPUT_IDS].squeeze() for item in tokenized_texts], dim=0)
    masks = torch.stack([item[ATTENTION_MASK].squeeze() for item in tokenized_texts], dim=0)
    labels = torch.tensor(labels)

    data = TensorDataset(indexed_texts, masks, labels)
    
    if sampling_type == params.RANDOM_SAMPLING:
        sampler = RandomSampler(data)
    elif sampling_type == params.SEQUENTIAL_SAMPLING:
        sampler = SequentialSampler(data)
    else:
        print('Wrong Sampling Type: ' + sampling_type)
        return None
        
    dataloader = DataLoader(data, sampler=sampler, batch_size=config[params.BATCH_SIZE])
    return dataloader

def get_data_loaders(input_df: pd.DataFrame,
                     config: dict, 
                     sampling_type: str,
                     tokenizer,
                     break_df_func) -> list:
    input_df[config[params.LENGTH_COL]] = input_df[config[params.TEXT_COL]].apply(lambda x: len(x.split()))
    df_list = break_df_func(input_df, config)
    dataloaders = []
    for df in df_list:
        dataloader = df_to_dataloader(df, tokenizer, config, sampling_type)
        dataloaders.append(dataloader)
        
    return dataloaders
