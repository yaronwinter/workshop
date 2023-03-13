import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from utils import config as params
from utils import gen_utils

def test(model: nn.Module, test_set: str, config: dict) -> tuple:
    df = pd.read_csv(test_set)
    dataloaders = gen_utils.get_data_loaders(df,
                                         model.get_w2v_model(),
                                         config,
                                         params.SEQUENTIAL_SAMPLING,
                                         True,
                                         gen_utils.break_df_by_len)
    
    corrects = 0
    evaluated = 0
    t0 = time.time()
    model.eval()
    for dataloader in dataloaders:
        for data in dataloader:
            ids, labels, _, _ = data
            with torch.no_grad():
                logits = model(ids)
            preds = torch.argmax(logits, dim=1)
            corrects += (preds == labels).sum().item()
            evaluated += ids.shape[0]
        
    return (corrects / evaluated), (time.time() - t0)
