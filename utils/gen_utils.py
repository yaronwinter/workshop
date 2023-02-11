# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:54:55 2020

@author: YaronWinter
"""
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
import pandas as pd
import utils.config as params
from gensim.models.keyedvectors import KeyedVectors as EmbedModel

RHO = 0.95
LENGTHS_FRAME = 12
MIN_OCCURRENCES_BY_FRAME = 500

def intersect_strings(s: str, u: str):
    l = [s, u]
    cv = CountVectorizer(max_df=100, min_df=0, analyzer='char', lowercase=True)
    x = cv.fit_transform(l)
    jdict = {}
    for let in cv.vocabulary_.keys():
        n = min(x[0, cv.vocabulary_[let]], x[1, cv.vocabulary_[let]])
        if n > 0:
            jdict[let] = n
            
    for x in jdict:
        print(x + ': ' + str(jdict[x]))
    return

def break_df_by_len(df: pd.DataFrame, config: dict) -> list:
    df[params.TOKENS_COL] = df[params.TEXT_COL].apply(lambda x: x.split(' '))
    df[params.LENGTH_COL] = df[params.TOKENS_COL].map(len)
    occurrence2len = dict(df[params.LENGTH_COL].value_counts())
    occurrences = sorted(occurrence2len.items())
    frame_len = int(len(occurrences) / LENGTHS_FRAME) + 1
    
    df_list = []
    start_frame = 0
    end_frame = frame_len
    while start_frame < len(occurrences):
        total_occurs, end_frame = get_end_frame(start_frame, end_frame, occurrences)
            
        if end_frame > len(occurrences):
            end_frame = len(occurrences)
            
        curr_df = df[df[params.LENGTH_COL] >= occurrences[start_frame][0]]
        curr_df = curr_df[curr_df[params.LENGTH_COL] <= occurrences[end_frame - 1][0]]
        
        df_list.append(curr_df)
        
        start_frame = end_frame
        end_frame += frame_len
        
    total_occurs = 0
    for curr_df in df_list:
        total_occurs += len(curr_df)
    return df_list

def get_end_frame(prev_start: int, prev_end: int, occurrences: list) -> tuple:
    total_occurs = sum([x[1] for x in occurrences[prev_start:prev_end]])
    
    if total_occurs >= MIN_OCCURRENCES_BY_FRAME:
        return total_occurs, prev_end
    
    total_left = sum([x[1] for x in occurrences[prev_start:]])
    if total_left < 2 * MIN_OCCURRENCES_BY_FRAME:
        end_frame = len(occurrences)
        return total_left, end_frame
    
    end_frame = prev_end
    while total_occurs < MIN_OCCURRENCES_BY_FRAME:
        total_occurs += occurrences[end_frame][1]
        end_frame += 1
        
    return total_occurs, (end_frame + 1)

def df_to_dataloader(df: pd.DataFrame, w2v_model: EmbedModel, config: dict) -> DataLoader:
    texts = df[params.TOKENS_COL].values.tolist()
    labels = df[params.LABEL_COL].values.tolist()
    max_len = df[params.TOKENS_COL].map(len).max()

    indexed_texts = []
    for sentence in texts:
        sentence += [params.PAD_LABEL] * (max_len - len(sentence))
        
        ids = [w2v_model.key_to_index[params.UNK_LABEL] if not w2v_model.has_index_for(word) else w2v_model.key_to_index[word] for word in sentence]
        indexed_texts.append(ids)
        
    inputs, labels = tuple(torch.tensor(data) for data in [indexed_texts, labels])
    
    data = TensorDataset(inputs, labels)
    
    sampling_type = config[params.sampling_type]
    if sampling_type == params.RANDOM_SAMPLING:
        sampler = RandomSampler(data)
    elif sampling_type == params.SEQUENTIAL_SAMPLING:
        sampler = SequentialSampler(data)
    else:
        print('Wrong Sampling Type: ' + sampling_type)
        return None
        
    dataloader = DataLoader(data, sampler=sampler, batch_size=config[params.BATCH_SIZE])
    return dataloader
    
def get_data_loaders(input_df: pd.DataFrame, w2v_model: EmbedModel, config: dict) -> list:
    df_list = break_df_by_len(input_df)
    dataloaders = []
    for df in df_list:
        dataloader = df_to_dataloader(df, w2v_model, config[params.BATCH_SIZE], config[params.SAMPLING_TYPE])
        dataloaders.append(dataloader)
        
    return dataloaders

def get_test_sample(df: pd.DataFrame, w2v_model: EmbedModel) -> tuple:
    print('extract texts & labels')
    texts = [x.split(' ') for x in df.review.values.tolist()]
    sentiments = df.sentiment.values.tolist()
    
    print('index texts')
    tensors = []
    labels = []
    for sentence, sentiment in zip(texts, sentiments):
        ids = [w2v_model.key_to_index[params.UNK_LABEL] if not w2v_model.has_index_for(word) else w2v_model.key_to_index[word] for word in sentence]
        tensors.append(torch.tensor(ids, dtype=torch.long).view(1,-1))
        labels.append(torch.tensor([sentiment], dtype=torch.long))
                
    return tensors, labels

def get_optimizer(opt_name: str, parameters, lr):
    optimizer = None
    if opt_name == params.ADADELATA_OPT:
        optimizer = optim.Adadelta(parameters,
                                   lr=lr,
                                   rho=RHO)
    elif opt_name == params.SGD_OPT:
        optimizer = optim.SGD(parameters, lr)
    elif opt_name == params.ADAM_OPT:
        optimizer = optim.Adam(parameters, lr=lr)
    else:
        print('Wrong optimizer name: ' + opt_name)
        
    return optimizer
    
    
def get_loss_function(func_name: str):
    loss_func = None
    if func_name == params.CROSS_ENTROP_LOSS:
        loss_func = nn.CrossEntropyLoss()
    elif func_name == params.BCE_LOSS:
        loss_func = nn.BCELoss()
    else:
        print('Wrong loss function name: ' + func_name)
        
    return loss_func
