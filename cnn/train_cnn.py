from cnn import cnn
import pandas as pd
import numpy as np
import random
import torch
import time
import copy
from utils import config as params
from utils import gen_utils
from tqdm import tqdm

class CNN_Trainer:
    def __init__(self):
        self.cnn_model = None
        self.opt_model = None

    def train(self, config: dict) -> tuple:
        print('train_cnn_nlp - start')
        log_file = open(config[params.LOG_FILE_NAME], "w", encoding="utf-8")

        train_df = pd.read_csv(config[params.TRAIN_SET])
        self.cnn_model = cnn.CNN(config, train_df[config[params.LABEL_COL]].unique().shape[0])
        
        print('set optimizer & loss')
        best_val_acc = 0
        best_epoch = -1
        self.opt_model = None
        
        optimizer = gen_utils.get_optimizer(config[params.OPTIMIZER_NAME],
                                        self.cnn_model.parameters(),
                                        config[params.LEARNING_RATE])
        loss_func = gen_utils.get_loss_function(config[params.LOSS_FUNCTION])
        
        print('get data loaders')
        sampling = params.RANDOM_SAMPLING if config[params.SEED_VALUE] < 0 else params.SEQUENTIAL_SAMPLING
        print('\tsampling = ' + sampling)
        dataloaders = gen_utils.get_data_loaders(train_df, 
                                             self.cnn_model.get_w2v_model(),
                                             config[params.BATCH_SIZE],
                                             sampling)
        
        num_epochs = config[params.NUM_EPOCHS]
        print('start training loops. #epochs = ' + str(num_epochs))
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*30)  
        
        log_file.write(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Acc':^9} | {'Elapsed':^9}\n")
        log_file.write("-"*30 + '\n')  
        log_file.flush()
            
        
        min_loss = 100
        num_no_imp = 0
        for i in range(num_epochs):
            epoch = i + 1
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0
            
            self.cnn_model.train()
            
            for dataloader in dataloaders:
                for step, batch in tqdm(enumerate(dataloader)):
                    ids, labels = tuple(t for t in batch)
                    
                    optimizer.zero_grad()
                    
                    logits = self.cnn_model(ids)
                    
                    loss = loss_func(logits, labels)
                    total_loss += loss.item()
                    num_batches += 1
                    
                    loss.backward()
                    
                    optimizer.step()
                
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            # Validation test.
            val_acc, val_time = test_cnn_nlp(self.cnn_model,
                                             config[params.VALIDATION_SET],
                                             config,
                                             False)
            val_acc *= 100
            print(f"{epoch:^7} | {avg_loss:^12.6f}  {val_acc:^9.2f} | {epoch_time:^9.2f}")
            log_file.write(f"{epoch:^7} | {avg_loss:^12.6f}  {val_acc:^9.2f} | {epoch_time:^9.2f}\n")
            log_file.flush()
                
            if avg_loss < min_loss:
                min_loss = avg_loss
                num_no_imp = 0
            else:
                num_no_imp += 1
                
            if num_no_imp > config[params.EARLY_STOP_MAX_NO_IMP]:
                print('early stop exit')
                log_file.write('\tEarly Stop exit\n')
                log_file.flush()
                break
            
            if epoch < config[params.MIN_VALID_EPOCHS]:
                continue
            
            if avg_loss > config[params.MAX_VALID_LOSS]:
                continue
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.opt_model = copy.deepcopy(self.cnn_model)
                best_epoch = epoch
        
        print('train_cnn_nlp - end')
        return self.cnn_model, self.opt_model, best_epoch

def test_cnn_nlp(model, test_set, config, print_mode=False):
    if print_mode:
        print('test cnn - start')
        
    df = pd.read_csv(test_set)
    dataloaders = gen_utils.get_data_loaders(df,
                                         model.get_w2v_model(),
                                         config[params.BATCH_SIZE],
                                         params.SEQUENTIAL_SAMPLING)
    
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

def end_train(last_model, opt_model, config, log_file):
    accuracy, run_time = test_cnn_nlp(last_model,
                                    config[params.TEST_SET],
                                    False)
    str_acc = "{:.5f}".format(accuracy)
    str_time = "{:.0f}".format(run_time)
    log_file.write('Last Model\t' + str_acc + '\t' + str_time + '\n')
    log_file.flush()
    print('Last Model\t' + str_acc + '\t' + str_time)
        
    # Print optimal
    opt_acc, run_time = test_cnn_nlp(opt_model,
                                    config[params.TEST_SET],
                                    False)
    val_acc, run_time = test_cnn_nlp(opt_model,
                                    config[params.VALIDATION_SET],
                                    False)

    test_acc = "test: {:.5f}".format(opt_acc)
    val_acc = "val: {:.5f}".format(val_acc)
    log_file.write(str(i+1) + ' (opt[' + str(opt_epoch) + ']): ' + val_acc + ', ' + test_acc + '\n')
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
