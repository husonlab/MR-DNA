import sys
import argparse
import time
import os
import pandas as pd
import numpy as np
import tokenizers
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from warmup_scheduler_pytorch import WarmUpScheduler
from torch.utils.data.distributed import DistributedSampler
import time
from transformers import AutoTokenizer, DistilBertConfig, DistilBertTokenizerFast, DistilBertForTokenClassification, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertModel
from transformers import AdamW, get_linear_schedule_with_warmup, DataCollatorForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from Bio.Seq import Seq
from tqdm.autonotebook import tqdm, trange
from ast import literal_eval
import utils
from models import DistilBertCRF_Focal, DistilBertCRF, DistilBertCRF_MethyLoss
import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MR-DNA-50', help='Dataset type')
    parser.add_argument('--status', type=str, default='test', help='training model or test model')
    parser.add_argument('--model', type=str, default='DistilBertCRF_MethyLoss', help='Model type')
    parser.add_argument('--savePath', type=str, default='result/model', help='path for saving the model in train mode')

    parser.add_argument('--lr', type=float, default=2e-5, help='Number of learning rate.')
    parser.add_argument('--epochs', type=int, default=64, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of batch size.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping (# of epochs).')
    parser.add_argument('--lr_decay', type=int, default=6, help='Patience for decreasing learning rate (# of epochs).')

    args = parser.parse_args()

    # Define the hyperparameters
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epoch
    lr_patience = args.lr_decay
    earlyStop_patience = args.patience

    length_map = {'MR-DNA-50': 50,
                  'MR-DNA-100': 100,
                  'MR-DNA-200': 200}
    sample_length = length_map[args.dataset]

    # define project directory
    root_dir = os.path.abspath(os.path.dirname(os.getcwd())) #../../MR-DNA/
    # read data and data preprocessing
    mydf = pd.read_csv(f'{root_dir}/database/{args.dataset}/{args.status}.txt', sep='\t', converters={'methy_pos': literal_eval})
    mydf = mydf.rename(columns={'sequence': 'text', 'methy_pos':'label'})
    mydf = mydf[['text', 'label']]
    mydf['label'] = list(map(lambda x: sorted(x), mydf['label']))
    # drop duplicate
    mydf['new_label'] = mydf['label'].apply(lambda x: ', '.join(map(str, x)))
    mydf = mydf.drop(['label'], axis=1)
    mydf = mydf.drop_duplicates(subset='text', keep='first')
    mydf['label'] = mydf['new_label'].apply(lambda x: x.split(', '))
    mydf['label'] = mydf['label'].apply(lambda x: [int(i) for i in x])
    mydf = mydf.drop(['new_label'], axis=1)

    mydf['sentence'] = list(map(lambda x: utils.seq2kmer_3mer(x, 3), mydf['text']))
    mydf['ner_tags'] = list(map(lambda x, y: utils.label_ner_3mer_3label(x, y), mydf['sentence'], mydf['label']))

    mydf = mydf[['ner_tags', 'sentence', 'label']]

    label_names = ['O', 'Methyl', 'Non-Methyl']
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    def str2num(seq):
        tmp_list = []
        for i in seq:
            tmp_list.append(label2id[i])
        return tmp_list

    mydf['label_ids'] = list(map(lambda x: str2num(x), mydf['ner_tags']))
    mydf['CG_anno'] = list(map(lambda x: utils.func_CG(x), mydf['sentence']))

    my_tokenizer, config_file, masked_language_model, pretrained_model_for_token_classification = [DistilBertTokenizerFast, DistilBertConfig, DistilBertForMaskedLM, DistilBertForTokenClassification]

    model_map = {'DistilBertCRF_MethyLoss': DistilBertCRF_MethyLoss,
                 'DistilBertCRF_Focal': DistilBertCRF_Focal,
                 'DistilBertCRF': DistilBertCRF}
    _model = model_map[args.model]


    if args.status == 'train':
        finetune_model_path = os.path.join(root_dir, args.savePath, args.dataset)
        if not os.path.exists(finetune_model_path):
            os.mkdir(finetune_model_path)
    else:
        finetune_model_path = os.path.join(root_dir, 'model', args.dataset)

    if args.status == 'train':
        # split data to training set and validation set
        tmp_train_dataset, tmp_valid_dataset = train_test_split(mydf[['sentence', 'ner_tags', 'label_ids', 'CG_anno']], test_size=0.2, random_state=22)
        tmp_train_dataset.reset_index(drop=True, inplace=True)
        tmp_valid_dataset.reset_index(drop=True, inplace=True)

    # load tokenizer and pretrained model
    tokenizer = DistilBertTokenizerFast.from_pretrained(f'./pretrained/tokenizer/5mC')
    distilbert = DistilBertForTokenClassification.from_pretrained('wenhuan/MuLan-Methyl-DistilBERT_5hmC', num_labels=3, ignore_mismatched_sizes=True)
    distilbert.resize_token_embeddings(len(tokenizer))

    # load model structure
    model = _model(distilbert, num_labels=3)

    if args.staus == 'train':
        # Create an instance of the TrainDataset
        train_dataset = utils.MyDataset(tmp_train_dataset, tokenizer, label=True)
        valid_dataset = utils.MyDataset(tmp_valid_dataset, tokenizer, label=True)
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = RandomSampler(valid_dataset)
        # Create the train DataLoader
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)
    else:
        tmp_test_dataset = mydf
        # Create an instance of the TestDataset
        test_dataset = utils.MyDataset(tmp_test_dataset, tokenizer)
        # Create the train DataLoader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.load_state_dict(torch.load(f'{finetune_model_path}/{args.dataset}_model.pth'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(100)

    # training
    if args.status == 'train':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, verbose=True) # min loss, max acc

        # Model training
        print(f'start model training')
        best_loss = float('inf')
        early_stopper = models.EarlyStopper(patience=earlyStop_patience, min_delta=0)
        epoch_tqdm = trange(num_epochs, desc="Epoch")
        model.to(device)
        model = nn.DataParallel(model)
        for epoch in epoch_tqdm:
            start_time = time.time()
            # Create an instance of the ensemble model
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids, attention_mask, CG_annotation, labels = batch
                input_ids, attention_mask, CG_annotation, labels = input_ids.to(device), attention_mask.to(device), CG_annotation.to(device), labels.to(device)

                if input_ids.shape[1] == sample_length:
                    pass
                else:
                    print('tokenizer size error')

                # zero the gradients
                optimizer.zero_grad()
                # Forward pass
                loss, _ = model(input_ids, attention_mask=attention_mask, CG_annotation=CG_annotation, labels=labels)
                loss = loss.mean()
                # Backward pass
                loss.backward()
                # update parameters
                optimizer.step()
                # Record the training loss
                total_loss = total_loss+loss.item()

            train_loss = total_loss/len(train_loader)
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {epoch+1}/{num_epochs} Loss: {train_loss:.4f} Training Time: {epoch_time} seconds")

            # model validation
            model.eval()
            model.to(device)

            all_predictions = []
            all_labels = []
            all_loss = 0.0

            with torch.no_grad():

                for _batch in valid_loader:

                    _input_ids, _attention_mask, _CG_annotation, _labels = _batch
                    _input_ids, _attention_mask, CG_annotation, _labels = _input_ids.to(device), _attention_mask.to(device), CG_annotation.to(device), _labels.to(device)

                    if _input_ids.shape[1] == sample_length:
                        pass
                    else:
                        print('tokenizer size error')
                    loss, predictions = model(_input_ids, attention_mask=_attention_mask, CG_annotation=_CG_annotation, labels=_labels) # loss, tags
                    loss = loss.mean()
                    all_loss = all_loss + loss.item()
                    all_predictions.extend(predictions)
                    all_labels.extend(_labels)

            valid_loss = all_loss/len(valid_loader)
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            evaluation_results = models.compute_metrics(all_labels.cpu().numpy(), all_predictions.cpu().numpy())
            _acc = evaluation_results['acc']
            valid_f1 = evaluation_results['f1']
            _recall = evaluation_results['recall']
            _precision = evaluation_results['precision']

            print(f'Epoch {epoch+1}: Validation loss: {valid_loss: .5f}, Validation Accuracy: {_acc: .4f}, f1: {valid_f1: .4f}, recall: {_recall: .4f}, precision: {_precision: .4f}')

            lr_scheduler.step(valid_loss)

            # save the best model
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.module.state_dict(), f'{finetune_model_path}/{epoch+1}_{valid_loss:.4f}_model.pth')

            # early stop
            if early_stopper.early_stop(valid_loss):
                print(f'early stopped at epoch {epoch+1}')
                print(f'Saving the last model')
                torch.save(model.module.state_dict(), f'{finetune_model_path}/{epoch+1}_{valid_loss:.4f}_model.pth')
                break

    else:
        model.eval()
        model.to(device)
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, CG_annotation, labels = batch
                input_ids, attention_mask, CG_annotation, labels = input_ids.to(device), attention_mask.to(device), CG_annotation.to(device), labels.to(device)

                if input_ids.shape[1] == sample_length:
                    pass
                else:
                    print('tokenizer size error')
                predictions = model(input_ids, attention_mask=attention_mask, CG_annotation=CG_annotation) # tags
                all_predictions.extend(predictions)
                all_labels.extend(labels)

            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            evaluation_results = models.compute_metrics(all_labels.detach().cpu().numpy(), all_predictions.detach().cpu().numpy())
            acc = evaluation_results['acc']
            f1 = evaluation_results['f1']
            recall = evaluation_results['recall']
            precision = evaluation_results['precision']
            cm = evaluation_results['confusion matrix']

            print(f'accuracy {acc: .4f} f1 {f1: .4f} precision {precision: .4f} recall {recall: .4f}\nconfusion matrix {cm}')




