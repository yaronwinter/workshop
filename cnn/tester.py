import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
from utils import config as params
from utils import gen_utils
from tqdm import tqdm

def test(model: nn.Module, test_set: str, config: dict) -> tuple:
    df = pd.read_csv(test_set)
    dataloaders = gen_utils.get_data_loaders(df,
                                         model.get_w2v_model(),
                                         config,
                                         params.SEQUENTIAL_SAMPLING)
    
    corrects = 0
    evaluated = 0
    t0 = time.time()
    model.eval()
    for dataloader in tqdm(dataloaders):
        for step, data in tqdm(enumerate(dataloader)):
            ids, labels = data
            with torch.no_grad():
                logits = model(ids)
            preds = torch.argmax(logits, dim=1)
            corrects += (preds == labels).sum().item()
            evaluated += ids.shape[0]
        
    accuracy = corrects / evaluated
    run_time = time.time() - t0
    
    print('#corrects = ' + str(corrects))
    print('#evalueted = ' + str(evaluated))
    print('Accuracy = {:.5f}'.format(accuracy))
    print('\ttime = {:.0f}'.format(run_time))
    print('test cnn - end')
    return accuracy, run_time
