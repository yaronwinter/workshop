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
from tqdm import tqdm

EPS_ADV = 0.1
VAT_EPS = 5

class EMLoss(nn.Module):
    def __init__(self):
        super(EMLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        b = b / x.shape[0]
        return b

def generate_vat_texts(embed_texts: torch.Tensor) -> torch.Tensor:
    vat_vecs = torch.normal(0, 1, size=embed_texts.size())
    square = torch.sum(vat_vecs ** 2, [2], keepdim=True)
    vat_texts = vat_vecs / torch.sqrt(square)
    vat_texts = embed_texts + EPS_ADV*vat_texts
    return vat_texts

def generate_vat_loss(embed_texts: torch.Tensor, lengths: torch.Tensor, model: LSTM) -> torch.Tensor:
    x = embed_texts.clone().detach().to(torch.float)
    vat_texts = generate_vat_texts(x)
    vat_texts.requires_grad = True
    vat_logits = model(vat_texts, lengths)
    reg_logits = model(x, lengths)

    reg_log_logits = F.log_softmax(reg_logits, dim=1)
    vat_log_logits = F.log_softmax(vat_logits, dim=1)

    loss = F.kl_div(vat_log_logits, reg_log_logits, reduction="batchmean", log_target=True)
    loss.backward()

    grad = vat_texts.grad
    grad = grad.clone().detach().to(torch.float)
    square = torch.sum(grad ** 2, [2], keepdim=True)
    norm_grad = grad / torch.sqrt(square)
    star_vecs = embed_texts.clone().detach().to(torch.float) + VAT_EPS*norm_grad
    star_logits = model(star_vecs, lengths)
    star_log_logits = F.log_softmax(star_logits, dim=1)

    x = x.clone().detach().to(torch.float)
    reg_logits = model(x, lengths)
    reg_log_logits = F.log_softmax(reg_logits, dim=1)

    return F.kl_div(star_log_logits, reg_log_logits, reduction="batchmean", log_target=True)

class Trainer:
    def __init__(self):
        pass

    def train(self, config: dict) -> tuple:
        print('lstm trainer - start')
        log_file = open(config[params.LOG_FILE_NAME], "w", encoding="utf-8")
        seed_value = config[params.SEED_VALUE]
        gen_utils.set_seed(seed_value)
            
        non_labeled_df = pd.read_csv(config[params.UNLABELED_SET])
        train_df = pd.read_csv(config[params.TRAIN_SET])
        val_df = pd.read_csv(config[params.VALIDATION_SET])
        test_df = pd.read_csv(config[params.TEST_SET])

        pending_model = LSTM(config=config, num_classes=train_df[config[params.LABEL_COL]].unique().shape[0])
        optimal_model = None
        
        optimizer =  gen_utils.get_optimizer(pending_model.parameters(), config=config)
        
        num_epochs = config[params.NUM_EPOCHS]
        min_epochs_to_stop = config[params.MIN_EPOCHS_TO_STOP]
        print('start training loops. #epochs = ' + str(num_epochs))
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Test Acc':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*50)  
        
        log_file.write(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Test Acc':^10} | {'Val Acc':^9} | {'Elapsed':^9}\n")
        log_file.write("-"*50 + "\n")
            
        
        best_val_acc = 0
        best_test_acc = 0
        best_val_epoch = -1
        best_test_epoch = -1
        min_loss = 100
        num_no_imp = 0
        em_loss_func = EMLoss()
        ml_loss_func = nn.CrossEntropyLoss()
        validation_dl = gen_utils.get_data_loaders(val_df, pending_model.w2v_model, config=config, sampling_type=params.SEQUENTIAL_SAMPLING, is_labeled=1, break_df_func=gen_utils.break_by_batch_size)
        test_dl = gen_utils.get_data_loaders(test_df, pending_model.w2v_model, config=config, sampling_type=params.SEQUENTIAL_SAMPLING,is_labeled=1, break_df_func=gen_utils.break_by_batch_size)
        train_dl = gen_utils.get_data_loaders(train_df, pending_model.w2v_model, config=config, sampling_type=params.SEQUENTIAL_SAMPLING, is_labeled=1, break_df_func=gen_utils.break_by_batch_size)
        non_labeled_dl = gen_utils.get_data_loaders(non_labeled_df, pending_model.w2v_model, config=config, sampling_type=params.SEQUENTIAL_SAMPLING, is_labeled=0, break_df_func=gen_utils.break_by_batch_size)
        train_set_dl = train_dl + non_labeled_dl
        print("type train set: " + str(type(train_set_dl)))
        print("len train set: " + str(len(train_set_dl)))
        print("len labeled train: " + str(len(train_dl)))
        print("len non labeled: " + str(len(non_labeled_dl)))
        for i in range(num_epochs):
            epoch = i + 1
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0

            random.shuffle(train_set_dl)
            pending_model.train()
            for dl in tqdm(train_set_dl):
                for texts, labels, lengths, is_labeled_indicators in dl:
                    optimizer.zero_grad()
                    embed_texts = pending_model.embed_text(texts)
                    ml_logits = pending_model(embed_texts, lengths)
                    if is_labeled_indicators[0] > 0:
                        ml_loss = ml_loss_func(ml_logits, labels)
                    else:
                        ml_loss = None

                    x = embed_texts.clone().detach().to(torch.float)
                    em_logits = pending_model(x, lengths)
                    em_loss = em_loss_func(em_logits)

                    vat_loss = generate_vat_loss(embed_texts=embed_texts, lengths=lengths, model=pending_model)

                    if ml_loss is not None:
                        ml_loss.backward()
                    em_loss.backward()
                    vat_loss.backward()
                    optimizer.step()

                    total_loss += em_loss.item()
                    total_loss += vat_loss.item()
                    if ml_loss is not None:
                        total_loss += ml_loss.item()
                    num_batches += 1

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
                
            if num_no_imp > config[params.EARLY_STOP_MAX_NO_IMP] and epoch > min_epochs_to_stop:
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
        
        print('train_lstm_nlp - end')
        print("Best test results: acc={:.3f}".format(best_test_acc) + ", epoch=" + str(best_test_epoch))
        print("Best val results: acc={:.3f}".format(best_val_acc) + ", epoch=" + str(best_val_epoch))
        log_file.write("Best test results: acc={:.3f}".format(best_test_acc) + ", epoch=" + str(best_test_epoch) + "\n")
        log_file.write("Best val results: acc={:.3f}".format(best_val_acc) + ", epoch=" + str(best_val_epoch) + "\n")
        log_file.flush()
        end_train(pending_model, optimal_model, test_dl, validation_dl, config, log_file)
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
