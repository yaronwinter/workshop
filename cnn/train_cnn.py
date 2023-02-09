# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 13:23:12 2020

@author: YaronWinter
"""
from cnn_nlp import CNN_NLP
import pandas as pd
import numpy as np
import random
import torch
import time
import copy
import utils

DROPOUT = 0.5
MAX_LOSS_FOR_BEST = 0.1
MIN_EPOCHS_FOR_BEST = 10

OUT_FILE_NAME = "/home/yaron/torch_env/ws/hiredscore/logs/cnn_imdb2.txt"
DATA_FOLDER = "/home/yaron/torch_env/ws/hiredscore/embed/data/"
FILTER_WIDTH = 100
FILTER_WINDOWS = [3,4,5]
FREEZE_W2V = True
LEARNING_RATE = 0.25
LOSS_FUNCTION = utils.CROSS_ENTROP_LOSS
NUM_EPOCHS = 20
NUM_LOOPS = 1
OPTIMIZER_NAME = utils.ADADELATA_OPT
SEED_VALUE = -1
TEST_SET = "imdb_test.csv"
TRAIN_SET = "imdb_train.csv"
VALIDATION_SET = "validate_set.csv"
W2V_MODEL = "/home/yaron/torch_env/ws/hiredscore/embed/model/w2v.model"

class CNN_Trainer:
    def __init__(self):
        self.cnn_model = None
        self.opt_model = None

    def train(self, out_file):
        print('train_cnn_nlp - start')
        
        print('allocate the model')
        if self.cnn_model is not None:
            del self.cnn_model
            
        train_df = pd.read_csv(DATA_FOLDER + TRAIN_SET)
        self.cnn_model = CNN_NLP(W2V_MODEL,
                                 train_df[utils.SENTIMENT_COL].nunique(),
                                 windows=FILTER_WINDOWS,
                                 width=FILTER_WIDTH,
                                 dropout=DROPOUT,
                                 freeze_embedding=FREEZE_W2V)
        
        print('set optimizer & loss')
        best_val_acc = 0
        best_epoch = -1
        self.opt_model = None
        
        optimizer = utils.get_optimizer(OPTIMIZER_NAME,
                                        self.cnn_model.parameters(),
                                        LEARNING_RATE)
        loss_func = utils.get_loss_function(LOSS_FUNCTION)
        
        print('get data loaders')
        sampling = utils.RANDOM_SAMPLING if SEED_VALUE < 0 else utils.SEQUENTIAL_SAMPLING
        print('\tsampling = ' + sampling)
        dataloaders = utils.get_data_loaders(train_df, 
                                             self.cnn_model.get_w2v_model(),
                                             utils.BATCH_SIZE,
                                             sampling)
        
        num_epochs = NUM_EPOCHS
        print('start training loops. #epochs = ' + str(num_epochs))
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*30)  
        
        if out_file is not None:
            out_file.write(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Acc':^9} | {'Elapsed':^9}\n")
            out_file.write("-"*30 + '\n')  
            out_file.flush()
            
        
        min_loss = 100
        num_no_imp = 0
        for i in range(num_epochs):
            epoch = i + 1
            t0_epoch = time.time()
            total_loss = 0
            num_batches = 0
            
            self.cnn_model.train()
            
            for dataloader in dataloaders:
                for step, batch in enumerate(dataloader):
                    ids, labels = tuple(t for t in batch)
                    
                    optimizer.zero_grad()
                    
                    logits = self.cnn_model(ids)
                    
                    loss = loss_func(logits, labels)
                    total_loss += loss.item()
                    num_batches += 1
                    
                    loss.backward()
                    
                    optimizer.step()
                
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - t0_epoch
            
            # Validation test.
            val_acc, val_time = test_cnn_nlp(self.cnn_model,
                                             VALIDATION_SET,
                                             False)
            val_acc *= 100
            print(f"{epoch:^7} | {avg_loss:^12.6f}  {val_acc:^9.2f} | {epoch_time:^9.2f}")
            if out_file is not None:
                out_file.write(f"{epoch:^7} | {avg_loss:^12.6f}  {val_acc:^9.2f} | {epoch_time:^9.2f}\n")
                out_file.flush()
                
            if avg_loss < min_loss:
                min_loss = avg_loss
                num_no_imp = 0
            else:
                num_no_imp += 1
                
            if num_no_imp > utils.EARLY_STOP_MAX_NO_IMP:
                print('early stop exit')
                out_file.write('\tEarly Stop exit\n')
                out_file.flush()
                break
            
            if epoch < MIN_EPOCHS_FOR_BEST:
                continue
            
            if avg_loss > MAX_LOSS_FOR_BEST:
                continue
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.opt_model = copy.deepcopy(self.cnn_model)
                best_epoch = epoch
        
        print('train_cnn_nlp - end')
        return self.cnn_model, self.opt_model, best_epoch

def test_cnn_nlp(model, test_set, print_mode=False):
    if print_mode:
        print('test cnn - start')
        
    df = pd.read_csv(DATA_FOLDER + test_set)
    dataloaders = utils.get_data_loaders(df,
                                         model.get_w2v_model(),
                                         utils.BATCH_SIZE,
                                         utils.SEQUENTIAL_SAMPLING)
    
    if print_mode:
        print('sample size = ' + str(len(dataloaders)))
        
    corrects = 0
    evaluated = 0
    t0 = time.time()
    model.eval()
    for dataloader in dataloaders:
        for step, data in enumerate(dataloader):
            ids, labels = data
            with torch.no_grad():
                logits = model(ids)
            preds = torch.argmax(logits, dim=1)
            corrects += (preds == labels).sum().item()
            evaluated += ids.shape[0]
           
        if print_mode:
            print('\t#Evalueted = ' + str(evaluated) + ', #Corrects=' + str(corrects))
        
    accuracy = corrects / evaluated
    run_time = time.time() - t0
    
    if print_mode:
        print('#corrects = ' + str(corrects))
        print('#evalueted = ' + str(evaluated))
        print('Accuracy = {:.5f}'.format(accuracy))
        print('\ttime = {:.0f}'.format(run_time))
        print('test cnn - end')
    return accuracy, run_time

def train_cnn_loop(loop_ind, out_file_name):
    print('working on loop: ' + str(loop_ind) + ' - ' + out_file_name)
    num_loops = NUM_LOOPS
    
    f = open(out_file_name, 'w', encoding='utf-8')
    
    accum_final_acc = 0
    accum_val_acc = 0
    accum_opt_acc = 0
    for i in range(num_loops):
        print('working on loop ' + str(i+1))
        trainer = CNN_Trainer()
        cnn_model, opt_model, opt_epoch = trainer.train(f)
        accuracy, run_time = test_cnn_nlp(cnn_model,
                                          TEST_SET,
                                          False)
        accum_final_acc += accuracy
        str_acc = "{:.5f}".format(accuracy)
        str_time = "{:.0f}".format(run_time)
        f.write(str(i+1) + ':\t' + str_acc + '\t' + str_time + '\n')
        f.flush()
        print(str(i+1) + ':\t' + str_acc + '\t' + str_time)
        
        # Print optimal
        opt_acc, run_time = test_cnn_nlp(opt_model,
                                         TEST_SET,
                                         False)
        val_acc, run_time = test_cnn_nlp(opt_model,
                                         VALIDATION_SET,
                                         False)

        accum_val_acc += val_acc
        accum_opt_acc += opt_acc

        test_acc = "test: {:.5f}".format(opt_acc)
        val_acc = "val: {:.5f}".format(val_acc)
        f.write(str(i+1) + ' (opt[' + str(opt_epoch) + ']): ' + val_acc + ', ' + test_acc + '\n')
        f.flush()
        print(str(i+1) + ' (opt[' + str(opt_epoch) + ']): ' + val_acc + ', ' + test_acc)
        
        
    final_acc = "Mean Final={:.5f}".format(accum_final_acc / num_loops)
    val_acc = "Mean Val={:.5f}".format(accum_val_acc / num_loops)
    opt_acc = "Mean Opt={:.5f}".format(accum_opt_acc / num_loops)
    f.write(final_acc + ', ' + val_acc + ', ' + opt_acc + '\n')
    print(final_acc + ', ' + val_acc + ', ' + opt_acc)
    
    f.flush()
    f.close()
    return opt_model


def train_cnn_loops(num_loops):
    for i in range(num_loops):
        if SEED_VALUE >= 0:
            random.seed(SEED_VALUE)
            np.random.seed(SEED_VALUE)
            torch.manual_seed(SEED_VALUE)
            torch.cuda.manual_seed_all(SEED_VALUE)
            
        train_cnn_loop(i, OUT_FILE_NAME)
    return
