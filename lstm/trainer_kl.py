import time
import copy
import torch
import torch.nn as nn
import random
import pandas as pd
from utils import config as params
from utils import gen_utils
from lstm.lstm import LSTM
from lstm import tester
import torch.nn.functional as F

EPS_ADV = 0.1
class Trainer:
    def __init__(self):
        pass

    def train(self, config: dict) -> tuple:
        print('latm trainer - start')
        log_file = open(config[params.LOG_FILE_NAME], "w", encoding="utf-8")
        seed_value = config[params.SEED_VALUE]
        gen_utils.set_seed(seed_value)
            
        train_df = pd.read_csv(config[params.TRAIN_SET])
        val_df = pd.read_csv(config[params.VALIDATION_SET])
        test_df = pd.read_csv(config[params.TEST_SET])

        pending_model = LSTM(config=config, num_classes=train_df[config[params.LABEL_COL]].unique().shape[0])
        optimal_model = None
        
        optimizer =  gen_utils.get_optimizer(pending_model.parameters(), config=config)
        log_soft_max = nn.LogSoftmax(dim=1)
        
        num_epochs = config[params.NUM_EPOCHS]
        print('start training loops. #epochs = ' + str(num_epochs))
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Test Acc':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*50)  
        
        log_file.write(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Test Acc':^10} | {'Val Acc':^9} | {'Elapsed':^9}\n")
        log_file.write("-"*50 + "\n")
            
        
        best_val_acc = 0
        best_epoch = -1
        min_loss = 100
        num_no_imp = 0
        validation_dl = gen_utils.get_data_loaders(val_df, pending_model.w2v_model, config=config, sampling_type=params.SEQUENTIAL_SAMPLING, break_df_func=gen_utils.break_by_batch_size)
        test_dl = gen_utils.get_data_loaders(test_df, pending_model.w2v_model, config=config, sampling_type=params.SEQUENTIAL_SAMPLING, break_df_func=gen_utils.break_by_batch_size)
        for i in range(num_epochs):
            print('start epoch')
            epoch = i + 1
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0

            train_dl = gen_utils.get_data_loaders(train_df, pending_model.w2v_model, config=config, sampling_type=params.SEQUENTIAL_SAMPLING, break_df_func=gen_utils.break_by_batch_size)
            random.shuffle(train_dl)
            pending_model.train()
            for dl in train_dl:
                print('start dl')
                for texts, labels, lengths in dl:
                    print("start batch")
                    optimizer.zero_grad()
                    embed_text = pending_model.embed_text(texts)

                    print("generate adv vecs")
                    adv_vecs = torch.normal(0, 1, size=embed_text.size())
                    square = torch.sum(adv_vecs ** 2, [2], keepdim=True)
                    adv_texts = adv_vecs / torch.sqrt(square)
                    adv_texts = embed_text + EPS_ADV*adv_texts
                    adv_texts.requires_grad = True

                    print('forward')
                    reg_logits = pending_model(embed_text, lengths)
                    adv_logits = pending_model(adv_texts, lengths)

                    print("convert to log probs")
                    reg_log_probs = log_soft_max(reg_logits)
                    adv_log_probs = log_soft_max(adv_logits)

                    print("KL Loss")
                    loss = F.kl_div(adv_log_probs, reg_log_probs, reduction="batchmean", log_target=True)
                    total_loss += loss.item()
                    num_batches += 1

                    print("backwards")
                    loss.backward()
                    print("texts size: " + str(embed_text.size()))
                    print("adv size: " + str(adv_texts.size()))
                    print("Grad:")
                    print("grad size: " + str(adv_texts.grad.size()))
                    print("Loss:")
                    print(loss)
                    optimizer.step()
                    break
                break
            break
                
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            # Validation test.
            val_acc, _ = tester.test(pending_model, validation_dl)
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
                optimal_model = copy.deepcopy(pending_model)
                best_epoch = epoch
        
        print('train_lstm_nlp - end')
        end_train(pending_model, optimal_model, test_dl, validation_dl, config, log_file)
        return pending_model, optimal_model, best_epoch

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
