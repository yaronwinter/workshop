import time
import copy
import torch
import random
import pandas as pd
from utils import config as params
from utils import gen_utils
from lstm.lstm import LSTM

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

        pending_model = LSTM(config=config, num_classes=train_df[config[params.LABEL_COL]].unique.shape[0])
        optimal_model = None

        best_val_acc = 0
        best_epoch = -1
        
        optimizer =  gen_utils.get_optimizer(self.lstm_model.parameters(), config=config)
        loss_func = gen_utils.get_loss_function(config[params.LOSS_FUNCTION])
        
        num_epochs = config[params.NUM_EPOCHS]
        print('start training loops. #epochs = ' + str(num_epochs))
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*50)  
        
        log_file.write(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Acc':^9} | {'Elapsed':^9}\n")
        log_file.write("-"*50 + "\n")
            
        
        min_loss = 100
        num_no_imp = 0
        validation_dl = gen_utils.get_data_loaders(val_df, pending_model.w2v_model, config=config, sampling_type=params.SEQUENTIAL_SAMPLING, break_df_func=gen_utils.break_df_by_len)
        test_dl = gen_utils.get_data_loaders(test_df, pending_model.w2v_model, config=config, sampling_type=params.SEQUENTIAL_SAMPLING, break_df_func=gen_utils.break_df_by_len)
        for i in range(num_epochs):
            epoch = i + 1
            t0_epoch = time.time()
            total_loss = 0

            train_dl = gen_utils.get_data_loaders(train_df, pending_model.w2v_model, config=config, sampling_type=config[params.SAMPLING_TYPE], break_df_func=gen_utils.break_by_batch_size)
            random.shuffle(train_dl)

            
            self.lstm_model.train()
            for dl in train_dl:

                if divmod(num_bat, 25)[1] == 0:
                    print('\tworking on batch: ' + str(num_bat) + ', len: ' + str(lengths[:5]))
                    
                if lengths[0] > MAX_TRAIN_LENGTH:
                    break

                optimizer.zero_grad()
                    
                logits = self.lstm_model(texts, lengths)
                
                loss = loss_func(logits, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                
            avg_loss = total_loss / len(train_iter)
            epoch_time = time.time() - t0_epoch
            
            # Validation test.
            val_acc, val_time = test_lstm(self.lstm_model, val_iter, False)
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
                self.opt_model = copy.deepcopy(self.lstm_model)
                best_epoch = epoch
        
        print('train_lstm_nlp - end')
        return self.lstm_model, self.opt_model, best_epoch
    
def test_lstm(model, test_iter, print_mode=False):
    if print_mode:
        print('#test batches = ' + str(len(test_iter)))
        
    if model is None:
        print('NULL model! Exit!')
        return 0,0
        
    corrects = 0
    evaluated = 0
    t0 = time.time()
    model.eval()
    for batch in test_iter:
        texts, lengths = batch.review
        labels = batch.sentiment

        with torch.no_grad():
            logits = model(texts, lengths)
        preds = torch.argmax(logits, dim=1)
        corrects += (preds == labels).sum().item()
        evaluated += texts.shape[0]
           
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


def train_lstm_loop(params):
    print('working on loop: ' + params[utils.OUT_FILE])
    f = open(utils.OUT_FOLDER + params[utils.OUT_FILE] + '.txt', 'w', encoding='utf-8')
    
    num_loops = params[utils.NUM_LOOPS]
    
    f.write('Run Train loops (' + str(num_loops) + ')\n')
    f.write('train set:\t' + params[utils.TRAIN_SET] + '\n')
    f.write('test set:\t' + params[utils.TEST_SET] + '\n')
    f.write('val set:\t' + params[utils.VALIDATION_SET] + '\n')
    f.write('w2v:\t' + params[utils.W2V_FILE] + '\n')
    f.write('lr:\t' + str(params[utils.LEARNING_RATE]) + '\n')
    f.write('opt:\t' + params[utils.OPTIMIZER_NAME] + '\n')
    f.write('freeze:\t' + str(params[utils.FREEZE_W2V]) + '\n')
    f.flush()
    
    print('load the data sets:')
    train_set = params[utils.TRAIN_SET]
    val_set = params[utils.VALIDATION_SET]
    test_set = params[utils.TEST_SET]
    w2v_file = params[utils.W2V_FILE]
    min_freq = params[utils.MIN_VOC_FREQ]
    TEXT, LABEL, fields = generate_fields(FIELDS_LIST)
    train_iter = load_bucket_iter(fields, TEXT, LABEL, min_freq, train_set, w2v_file)
    val_iter = load_bucket_iter(fields, TEXT, LABEL, min_freq, val_set, None)
    test_iter = load_bucket_iter(fields, TEXT, LABEL, min_freq, test_set, None)
    print('\tw2v vectors: ' + str(TEXT.vocab.vectors.shape))
        
    accum_final_acc = 0
    accum_val_acc = 0
    accum_opt_acc = 0
    for i in range(num_loops):
        print('working on loop ' + str(i+1))
        trainer = LSTM_Trainer()
        lstm_model, opt_model, opt_epoch = trainer.train(params,
                                                         f,
                                                         TEXT,
                                                         LABEL,
                                                         train_iter,
                                                         val_iter)
        accuracy, run_time = test_lstm(lstm_model, test_iter, False)
        accum_final_acc += accuracy
        str_acc = "{:.5f}".format(accuracy)
        str_time = "{:.0f}".format(run_time)
        f.write(str(i+1) + ':\t' + str_acc + '\t' + str_time + '\n')
        f.flush()
        print(str(i+1) + ':\t' + str_acc + '\t' + str_time)
        
        # Print optimal
        opt_acc, run_time = test_lstm(opt_model, test_iter, False)
        val_acc, run_time = test_lstm(opt_model, val_iter, False)

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
    return
