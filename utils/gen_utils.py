# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:54:55 2020

@author: YaronWinter
"""
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset

DATA_FOLDER = 'C:/WorkEnv/data_sets/ACLIMDB/'
LOGS_FOLDER = 'C:/WorkEnv/Spyder/Logs/'
IMDB_TRAIN_FOLDER = 'C:/WorkEnv/data_sets/ACLIMDB/aclimdb/train/'
IMDB_TEST_FOLDER = 'C:/WorkEnv/data_sets/ACLIMDB/aclimdb/test/'
ELEC_FOLDER = 'C:/WorkEnv/data_sets/Elec/'

TEXT_FIELD_NAME = 'review'
LABEL_FIELD_NAME = 'sentiment'
W2V_3G_MC7_SG_UG_NOHEAD = 'w2v_mc7_3g_d200_nohead.txt'
W2V_3G_MC7_SG_UG_WITHHEAD = 'w2v_mc7_3g_d200.txt'
IMDB_TRAIN_SET = 'imdb_train_set.csv'
IMDB_VAL_SET = 'imdb_val_set.csv'
IMDB_TEST_SET = 'imdb_test.csv'
ROTTEN_TOMAT_TRAIN_SET = 'rt_train_set.csv'
ROTTEN_TOMAT_VAL_SET = 'rt_val_set.csv'
ROTTEN_TOMAT_TEST_SET = 'rotten_tomatos_test.csv'
RANDOM_SAMPLING = 'random'
SEQUENTIAL_SAMPLING = 'sequential'
LENGTHS_FRAME = 12
MIN_OCCURRENCES_BY_FAME = 500
TOKENS_COL = 'tokens'
LENGTHS_COL = 'lengths'
REVIEW_COL = 'review'
SENTIMENT_COL = 'sentiment'
PAD_LABEL = '<pad>'
UNK_LABEL = '<unk>'

EARLY_STOP_MAX_NO_IMP = 2
RHO = 0.95
BATCH_SIZE = 50
ADADELATA_OPT = 'adadelta'
SGD_OPT = 'sgd'
ADAM_OPT = 'adam'
CROSS_ENTROP_LOSS = 'cross_entropy_loss'
BCE_LOSS = 'bce_loss'

OUT_FOLDER = 'C:/WorkEnv/Spyder/Logs/'
TRAIN_SET = 'train_set'
VALIDATION_SET = 'validation_set'
TEST_SET = 'test_set'
W2V_FILE = 'w2v_file'
NUM_EPOCHS = 'num_epochs'
LEARNING_RATE = 'lr'
FILTER_WINDOWS = 'windows'
FILTER_WIDTH = 'width'
OPTIMIZER_NAME = 'opt_name'
LOSS_FUNCTION = 'loss_func'
FREEZE_W2V = 'freeze_w2v'
OUT_FILE = 'out_file'
NUM_LOOPS = 'num_loops'
SEED_VALUE = 'seed_value'
MIN_VOC_FREQ = 'min_voc_freq'

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

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    
def load_cifar_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes


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
