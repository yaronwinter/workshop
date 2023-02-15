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
from utils.embedding import Embedded_Words
import copy
import random

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

def break_by_batch_size(df: pd.DataFrame, config: dict) -> list:
    sorted_df = df.sort_values(by=config[params.LENGTH_COL], axis=0, ascending=False, ignore_index=True)
    tokens = sorted_df[config[params.TOKENS_COL]].to_list()
    labels = sorted_df[config[params.LABEL_COL]].to_list()
    lengths = sorted_df[config[params.LENGTH_COL]].to_list()
    
    df_list = []
    batch_size = config[params.MINI_BATCH_SIZE]
    max_valid_length = config[params.MAX_TRAIN_LENGTH]
    header = {config[params.TOKENS_COL]:[], config[params.LABEL_COL]:[], config[params.LENGTH_COL]:[]}
    row_index = 0
    num_rows = len(labels)
    while row_index < num_rows:
        num_words = 0
        curr_df = copy.deepcopy(header)
        while num_words < batch_size:
            actual_length = min(max_valid_length, lengths[row_index])
            num_words += actual_length
            curr_df[config[params.TOKENS_COL]].append(tokens[row_index][:max_valid_length])
            curr_df[config[params.LABEL_COL]].append(labels[row_index])
            curr_df[config[params.LENGTH_COL]].append[lengths[row_index]]
            row_index += 1
        
        df_list.append(pd.DataFrame(curr_df))
        
    return df_list

def break_df_by_len(df: pd.DataFrame, config: dict) -> list:
    occurrence2len = dict(df[config[params.LENGTH_COL]].value_counts())
    occurrences = sorted(occurrence2len.items())
    frame_len = int(len(occurrences) / config[params.NUM_LENGTH_FRAMES]) + 1
    
    df_list = []
    start_frame = 0
    end_frame = frame_len
    while start_frame < len(occurrences):
        total_occurs, end_frame = get_end_frame(start_frame, end_frame, occurrences, config)
            
        if end_frame > len(occurrences):
            end_frame = len(occurrences)
            
        curr_df = df[df[config[params.LENGTH_COL]] >= occurrences[start_frame][0]]
        curr_df = curr_df[curr_df[config[params.LENGTH_COL]] <= occurrences[end_frame - 1][0]]
        
        df_list.append(curr_df)
        
        start_frame = end_frame
        end_frame += frame_len
        
    return df_list

def get_end_frame(prev_start: int, prev_end: int, occurrences: list, config: dict) -> tuple:
    total_occurs = sum([x[1] for x in occurrences[prev_start:prev_end]])
    
    if total_occurs >= config[params.MIN_FRAME_ITEMS]:
        return total_occurs, prev_end
    
    total_left = sum([x[1] for x in occurrences[prev_start:]])
    if total_left < 2 * config[params.MIN_FRAME_ITEMS]:
        end_frame = len(occurrences)
        return total_left, end_frame
    
    end_frame = prev_end
    while total_occurs < config[params.MIN_FRAME_ITEMS]:
        total_occurs += occurrences[end_frame][1]
        end_frame += 1
        
    return total_occurs, (end_frame + 1)

def df_to_dataloader(df: pd.DataFrame, w2v_model: Embedded_Words, config: dict, sampling_type: str) -> DataLoader:
    texts = df[config[params.TOKENS_COL]].values.tolist()
    labels = df[config[params.LABEL_COL]].values.tolist()
    lengths = df[config[params.LENGTH_COL]].to_list()
    max_len = df[config[params.TOKENS_COL]].map(len).max()

    indexed_texts = []
    drop_word_prob = config[params.DROP_WORD]
    for sentence in texts:
        sentence += [params.PAD_LABEL] * (max_len - len(sentence))
        ids = []
        for word in sentence:
            if word not in w2v_model.w2i:
                ids.append(w2v_model.w2i[params.UNK_LABEL])
            elif np.random.random() < drop_word_prob:
                ids.append(w2v_model.w2i[params.UNK_LABEL])
            else:
                ids.append(w2v_model.w2i[word])
        
        indexed_texts.append(ids)
        
    inputs, labels, lengths = tuple(torch.tensor(data) for data in [indexed_texts, labels, lengths])
    
    data = TensorDataset(inputs, labels)
    
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
                     w2v_model: Embedded_Words,
                     config: dict, 
                     sampling_type: str,
                     break_df_func) -> list:
    input_df[config[params.TOKENS_COL]] = input_df[config[params.TEXT_COL]].apply(lambda x: x.split(' '))
    input_df[config[params.LENGTH_COL]] = input_df[config[params.TOKENS_COL]].map(len)
    df_list = break_df_func(input_df, config)
    dataloaders = []
    for df in df_list:
        dataloader = df_to_dataloader(df, w2v_model, config, sampling_type)
        dataloaders.append(dataloader)
        
    return dataloaders

def get_test_sample(df: pd.DataFrame, w2v_model: Embedded_Words) -> tuple:
    print('extract texts & labels')
    texts = [x.split(' ') for x in df.review.values.tolist()]
    sentiments = df.sentiment.values.tolist()
    
    print('index texts')
    tensors = []
    labels = []
    for sentence, sentiment in zip(texts, sentiments):
        ids = [w2v_model.key_to_index[params.UNK_LABEL] if word not in w2v_model.w2i else w2v_model.w2i[word] for word in sentence]
        tensors.append(torch.tensor(ids, dtype=torch.long).view(1,-1))
        labels.append(torch.tensor([sentiment], dtype=torch.long))
                
    return tensors, labels

def get_optimizer(parameters, config: dict):
    optimizer = None
    opt_name = config[params.OPTIMIZER_NAME]
    if opt_name == params.ADADELATA_OPT:
        optimizer = optim.Adadelta(parameters,
                                   lr=config[params.LEARNING_RATE],
                                   rho=config[params.RHO])
    elif opt_name == params.SGD_OPT:
        optimizer = optim.SGD(parameters, config[params.LEARNING_RATE])
    elif opt_name == params.ADAM_OPT:
        optimizer = optim.Adam(parameters,
                               lr=config[params.LEARNING_RATE],
                               betas=(config[params.BETA_ONE],config[params.BETA_TWO],),
                               eps=config[params.ADAM_EPS])
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


def set_seed(seed_value: int):
    if seed_value >= 0:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
