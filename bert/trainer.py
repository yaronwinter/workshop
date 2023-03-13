import time
import copy
import torch.nn as nn
import random
import pandas as pd
from utils import config as params
from utils import gen_utils
from bert import utils as bert_utils
from bert import tester
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import AutoConfig
from tqdm import tqdm

class Trainer:
    def __init__(self):
        pass

    def train(self, config: dict) -> tuple:
        print('bert trainer - start')
        log_file = open(config[params.LOG_FILE_NAME], "w", encoding="utf-8")
        seed_value = config[params.SEED_VALUE]
        gen_utils.set_seed(seed_value)

        print("Load BERT model")
        print("\tConfig:")
        start_time = time.time()
        bert_config = AutoConfig.from_pretrained(config[params.BERT_CONFIG])
        label2id = config[params.BERT_LABELS2ID]
        bert_config.label2id = copy.deepcopy(label2id)
        bert_config.id2label = copy.deepcopy({label2id[x]:x for x in label2id})
        print("\tload time = {:.2f}".format(time.time() - start_time))

        print("\tTokenizer:")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(config[params.BERT_CONFIG])
        print("\tload time = {:.2f}".format(time.time() - start_time))

        print("\tModel:")
        start_time = time.time()
        pending_model = AutoModelForSequenceClassification.from_config(bert_config)
        optimal_model = None
        print("\tload time = {:.2f}".format(time.time() - start_time))
            
        print("load data loaders")
        train_df = pd.read_csv(config[params.TRAIN_SET])
        val_df = pd.read_csv(config[params.VALIDATION_SET])
        test_df = pd.read_csv(config[params.TEST_SET])

        train_dl = bert_utils.get_data_loaders(train_df, config, params.RANDOM_SAMPLING, tokenizer, gen_utils.break_by_batch_size)
        val_dl = bert_utils.get_data_loaders(val_df, config, params.SEQUENTIAL_SAMPLING, tokenizer, gen_utils.break_by_batch_size)
        test_dl = bert_utils.get_data_loaders(test_df, config, params.SEQUENTIAL_SAMPLING, tokenizer, gen_utils.break_by_batch_size)
        
        optimizer =  gen_utils.get_optimizer(pending_model.parameters(), config=config)
        loss_func = gen_utils.get_loss_function(config[params.LOSS_FUNCTION])
        
        num_epochs = config[params.NUM_EPOCHS]
        print('start training loops. #epochs = ' + str(num_epochs))
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Test Acc':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*50)  
        
        log_file.write(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Test Acc':^10} | {'Val Acc':^9} | {'Elapsed':^9}\n")
        log_file.write("-"*50 + "\n")
            
        
        best_val_acc = 0
        best_val_epoch = -1
        best_test_acc = 0
        best_test_epoch = -1
        min_loss = 100
        num_no_imp = 0
        for i in tqdm(range(num_epochs)):
            epoch = i + 1
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0

            random.shuffle(train_dl)
            pending_model.train()
            for dl in tqdm(train_dl):
                for indexed_texts, masks, labels in dl:
                    optimizer.zero_grad()
                    res = pending_model(input_ids=indexed_texts, attention_mask=masks)
                    loss = loss_func(res.logits, labels)
                    total_loss += loss.item()
                    num_batches += 1
                    loss.backward()
                    optimizer.step()
                
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            # Validation test.
            val_acc, _ = tester.test(pending_model, val_dl)
            train_acc, _ = tester.test(pending_model, train_dl)
            test_acc, _ = tester.test(pending_model, test_dl)
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
                
            if num_no_imp > config[params.EARLY_STOP_MAX_NO_IMP] and epoch > config[params.MIN_EPOCHS_TO_STOP]:
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
                optimal_model = copy.deepcopy(pending_model)
                best_val_epoch = epoch

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_epoch = epoch
        
        print('bert trainer - end')
        print("Best Val Acc = {:.2f}".format(best_val_acc) + ", Best Val Epoch = " + str(best_val_epoch))
        print("Best Test Acc = {:.2f}".format(best_test_acc) + ", Best Test Epoch = " + str(best_test_epoch))
        log_file.write("Best Val Acc = {:.2f}".format(best_val_acc) + ", Best Val Epoch = " + str(best_val_epoch) + "\n")
        log_file.writable("Best Test Acc = {:.2f}".format(best_test_acc) + ", Best Test Epoch = " + str(best_test_epoch) + "\n")
        end_train(pending_model, optimal_model, test_dl, val_dl, config, log_file)

        log_file.flush()
        log_file.close()

        return pending_model, optimal_model, best_val_epoch

def end_train(last_model: nn.Module, opt_model: nn.Module, test_dl: list, val_dl: list, config: dict, log_file):
    accuracy, run_time = tester.test(last_model, test_dl)
    str_acc = "{:.5f}".format(accuracy)
    str_time = "{:.1f}".format(run_time)
    log_file.write('Last Model\t' + str_acc + '\t' + str_time + '\n')
    log_file.flush()
    print('Last Model\t' + str_acc + '\t' + str_time)
        
    # Print optimal
    opt_acc, run_time = tester.test(opt_model, test_dl)
    val_acc, run_time = tester.test(opt_model, val_dl)

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
