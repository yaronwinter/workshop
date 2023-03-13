import time
import torch
import torch.nn as nn

def test(model: nn.Module, dataloaders: list):
    corrects = 0
    evaluated = 0
    start_time = time.time()
    model.eval()
    for dl in dataloaders:
        for texts, masks, labels in dl:
            with torch.no_grad():
                res = model(input_ids=texts, attention_mask=masks)
            preds = torch.argmax(res.logits, dim=1)
            corrects += (preds == labels).sum().item()
            evaluated += texts.shape[0]
        
    return (corrects / evaluated), (time.time() - start_time)
