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


RHO = 0.95

def intersect_strings(s, u):
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

def numpy_to_tensor(v):
    u = torch.zeros(v.shape[0])
    for i in range(v.shape[0]):
        u[i] = v[i]
        
    return u

def break_df_by_len(df, print_mode=False):
    if print_mode:
        print('break_df_by_len: len(df)=' + str(len(df)))
        
    df[TOKENS_COL] = df[REVIEW_COL].apply(lambda x: x.split(' '))
    df[LENGTHS_COL] = df[TOKENS_COL].map(len)
    occurrence2len = dict(df[LENGTHS_COL].value_counts())
    occurrences = sorted(occurrence2len.items())
    frame_len = int(len(occurrences) / LENGTHS_FRAME) + 1
    
    df_list = []
    start_frame = 0
    end_frame = frame_len
    while start_frame < len(occurrences):
        total_occurs, end_frame = get_end_frame(start_frame, end_frame, occurrences)
            
        if end_frame > len(occurrences):
            end_frame = len(occurrences)
        
        if print_mode:
            print('start = ' + str(start_frame) + ' [' + str(occurrences[start_frame][0]) + '], end = ' + str(end_frame) + ' [' + str(occurrences[end_frame-1][0]) + '], #occurs = ' + str(total_occurs))
            
        curr_df = df[df[LENGTHS_COL] >= occurrences[start_frame][0]]
        curr_df = curr_df[curr_df[LENGTHS_COL] <= occurrences[end_frame - 1][0]]
        
        df_list.append(curr_df)
        
        start_frame = end_frame
        end_frame += frame_len
        
    total_occurs = 0
    for curr_df in df_list:
        total_occurs += len(curr_df)
    return df_list

def get_end_frame(prev_start, prev_end, occurrences):
    total_occurs = sum([x[1] for x in occurrences[prev_start:prev_end]])
    
    if total_occurs >= MIN_OCCURRENCES_BY_FAME:
        return total_occurs, prev_end
    
    total_left = sum([x[1] for x in occurrences[prev_start:]])
    if total_left < 2 * MIN_OCCURRENCES_BY_FAME:
        end_frame = len(occurrences)
        return total_left, end_frame
    
    end_frame = prev_end
    while total_occurs < MIN_OCCURRENCES_BY_FAME:
        total_occurs += occurrences[end_frame][1]
        end_frame += 1
        
    return total_occurs, (end_frame + 1)

def df_to_dataloader(df, w2v_model, batch_size, sampling_type):
    texts = df[TOKENS_COL].values.tolist()
    labels = df[SENTIMENT_COL].values.tolist()
    max_len = df[TOKENS_COL].map(len).max()

    indexed_texts = []
    for sentence in texts:
        sentence += [PAD_LABEL] * (max_len - len(sentence))
        
        ids = [w2v_model.key_to_index[UNK_LABEL] if not w2v_model.has_index_for(word) else w2v_model.key_to_index[word] for word in sentence]
        indexed_texts.append(ids)
        
    inputs, labels = tuple(torch.tensor(data) for data in [indexed_texts, labels])
    
    data = TensorDataset(inputs, labels)
    
    if sampling_type == RANDOM_SAMPLING:
        sampler = RandomSampler(data)
    elif sampling_type == SEQUENTIAL_SAMPLING:
        sampler = SequentialSampler(data)
    else:
        print('Wrong Sampling Type: ' + sampling_type)
        return None
        
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader
    
def get_data_loaders(input_df, w2v_model, batch_size, sampling_type, print_mode=False):
    if print_mode:
        print('break the input df into segments')
        
    df_list = break_df_by_len(input_df)
    
    if print_mode:
        print('get_data_loaders: #df = ' + str(len(df_list)))
        
    dataloaders = []
    for df in df_list:
        dataloader = df_to_dataloader(df, w2v_model, batch_size, sampling_type)
        dataloaders.append(dataloader)
        
    return dataloaders

def get_test_sample(df, w2v_model):
    print('extract texts & labels')
    texts = [x.split(' ') for x in df.review.values.tolist()]
    sentiments = df.sentiment.values.tolist()
    
    print('index texts')
    tensors = []
    labels = []
    for sentence, sentiment in zip(texts, sentiments):
        ids = [w2v_model.key_to_index[UNK_LABEL] if not w2v_model.has_index_for(word) else w2v_model.key_to_index[word] for word in sentence]
        tensors.append(torch.tensor(ids, dtype=torch.long).view(1,-1))
        labels.append(torch.tensor([sentiment], dtype=torch.long))
                
    return tensors, labels

def get_optimizer(opt_name, parameters, lr):
    optimizer = None
    if opt_name == ADADELATA_OPT:
        optimizer = optim.Adadelta(parameters,
                                   lr=lr,
                                   rho=RHO)
    elif opt_name == SGD_OPT:
        optimizer = optim.SGD(parameters, lr)
    elif opt_name == ADAM_OPT:
        optimizer = optim.Adam(parameters, lr=lr)
    else:
        print('Wrong optimizer name: ' + opt_name)
        
    return optimizer
    
    
def get_loss_function(func_name):
    loss_func = None
    if func_name == CROSS_ENTROP_LOSS:
        loss_func = nn.CrossEntropyLoss()
    elif func_name == BCE_LOSS:
        loss_func = nn.BCELoss()
    else:
        print('Wrong loss function name: ' + func_name)
        
    return loss_func
