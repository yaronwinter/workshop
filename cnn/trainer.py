import torch.nn as nn
from cnn import cnn
from cnn import tester
import pandas as pd
import numpy as np
import random
import torch
import time
import copy
from utils import config as params
from utils import gen_utils

class Trainer:
    def __init__(self):
        pass

    def train(self, config: dict) -> tuple:
        print('cnn trainer - start')
        log_file = open(config[params.LOG_FILE_NAME], "w", encoding="utf-8")
        seed_value = config[params.SEED_VALUE]
        gen_utils.set_seed(seed_value)

        train_df = pd.read_csv(config[params.TRAIN_SET])
        pending_model = cnn.CNN(config, train_df[config[params.LABEL_COL]].unique().shape[0])
        optimal_model = None
        
        print('set optimizer & loss')
        best_val_acc = 0
        best_val_epoch = -1
        best_test_acc = 0
        best_test_epoch = -1
        optimizer = gen_utils.get_optimizer(pending_model.parameters(), config)
        loss_func_name = config[params.LOSS_FUNCTION]
        loss_func = gen_utils.get_loss_function(loss_func_name)
        
        print('get data loaders')
        sampling = params.RANDOM_SAMPLING if seed_value < 0 else params.SEQUENTIAL_SAMPLING
        print('\tsampling = ' + sampling)
        dataloaders = gen_utils.get_data_loaders(train_df, 
                                             pending_model.get_w2v_model(),
                                             config,
                                             config[params.SAMPLING_TYPE],
                                             True,
                                             gen_utils.break_df_by_len)
        
        num_epochs = config[params.NUM_EPOCHS]
        print('start training loops. #epochs = ' + str(num_epochs))
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Test Acc':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*50)  
        
        log_file.write(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Test Acc':^10} | {'Val Acc':^9} | {'Elapsed':^9}\n")
        log_file.write("-"*50 + "\n")
        log_file.flush()
            
        
        min_loss = 100
        num_no_imp = 0
        for i in range(num_epochs):
            epoch = i + 1
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0
            
            pending_model.train()
            
            for dataloader in dataloaders:
                for batch in dataloader:
                    ids, labels, _, _ = tuple(t for t in batch)
                    
                    optimizer.zero_grad()
                    
                    logits = pending_model(ids)
                    
                    loss = loss_func(logits, labels)
                    total_loss += loss.item()
                    num_batches += 1
                    
                    loss.backward()
                    
                    optimizer.step()
                
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            # Validation test.
            val_acc, _ = tester.test(pending_model, config[params.VALIDATION_SET], config)
            train_acc, _ = tester.test(pending_model, config[params.TRAIN_SET], config)
            test_acc, _ = tester.test(pending_model, config[params.TEST_SET], config)
            val_acc *= 100
            train_acc *= 100
            test_acc *= 100
            print(f"{epoch:^7} | {avg_loss:^12.6f} | {train_acc:^9.2f} | {test_acc:^9.2f} |  {val_acc:^9.4f} | {epoch_time:^9.2f}")
            log_file.write(f"{epoch:^7} | {avg_loss:^12.6f}  {train_acc:^9.2f} | {test_acc:^9.2f} |  {val_acc:^9.4f} | {epoch_time:^9.2f}\n")
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
                best_val_epoch = epoch
                optimal_model = copy.deepcopy(pending_model)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_epoch = epoch
        
        print('train_cnn_nlp - end')
        print("Best Val Acc = {:.2f}".format(best_val_acc) + ", best epoch = " + str(best_val_epoch))
        print("Best Test Acc = {:.2f}".format(best_test_acc) + ", best epoch = " + str(best_test_epoch))
        log_file.write("Best Val Acc = {:.2f}".format(best_val_acc) + ", best epoch = " + str(best_val_epoch) + "\n")
        log_file.write("Best Test Acc = {:.2f}".format(best_test_acc) + ", best epoch = " + str(best_test_epoch) + "\n")
        end_train(pending_model, optimal_model, config, log_file)
        log_file.close()
        return pending_model, optimal_model, best_val_epoch


def end_train(last_model: nn.Module, opt_model: nn.Module, config: dict, log_file):
    accuracy, run_time = tester.test(last_model, config[params.TEST_SET], config)
    str_acc = "{:.5f}".format(accuracy)
    str_time = "{:.0f}".format(run_time)
    log_file.write('Last Model\t' + str_acc + '\t' + str_time + '\n')
    log_file.flush()
    print('Last Model\t' + str_acc + '\t' + str_time)
        
    # Print optimal
    opt_acc, run_time = tester.test(opt_model, config[params.TEST_SET], config)
    val_acc, run_time = tester.test(opt_model, config[params.VALIDATION_SET], config)

    test_acc = "test: {:.5f}".format(opt_acc)
    val_acc = "val: {:.5f}".format(val_acc)
    log_file.write('Optimal Model\tTest=' + test_acc + '\tVal=' + val_acc + '\n')
    log_file.flush()
    print('Optimal Model\tTest=' + test_acc + '\tVal=' + val_acc)
    log_file.write("\nConfiguraton:\n")
    sorted_params = sorted(config.items())
    for param in sorted_params:
        log_file.write(param[0] + "\t" + str(param[1]) + "\n")
        log_file.flush()
