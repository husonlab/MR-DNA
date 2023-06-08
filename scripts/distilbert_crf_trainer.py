'''
this program is for training MR-DNA on the pocessed MR-DNA-50 dataset
'''
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(curPath))
#sys.path.append('/mnt/volume/project/5mC/github/scripts')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from transformers import DistilBertTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup, DataCollatorForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef, confusion_matrix
from Bio.Seq import Seq
import process_data
from distilbert_crf import DistilBertCRF_Focal
from ast import literal_eval

#dataPath = '/mnt/volume/project/5mC/github/database/MR-DNA-50/train.txt'
dataPath = '../database/MR-DNA-50/train.txt'
train_set = pd.read_csv(dataPath, sep='\t', converters={'label': literal_eval})

train_set['sentence'] = list(map(lambda x: process_data.seq2kmer_3mer(x, 3), train_set['text']))
train_set['ner_tags'] = list(map(lambda x, y: process_data.label_ner_3mer_3label(x, y), train_set['sentence'], train_set['label']))

train_df = train_set[['ner_tags', 'sentence', 'label']]

# statistics new tokens (contains N)
train_set['sentence'] = list(map(lambda x: process_data.seq2kmer_3mer(x, 3), train_set['text']))
tmp_kmer_list = ' '.join(list(train_set['sentence']))
uniq_kmer_list = list(set(tmp_kmer_list.split(' ')))
new_token = []
for ele in uniq_kmer_list:
    if 'N' in ele:
        new_token.append(ele)
# list all possible 3 mer permutations (in case some may not yet included in tokenizer\'s vocabulary)
def kmer_permutation(list_):
    res = []
    for i in list_:
        for j in list_:
            for k in list_:
                ele = f'{i}{j}{k}'
                res.append(ele)
    return res

list_3mer = kmer_permutation(['A','T','G','C'])

label_names = ['O', 'Methyl', 'Non-Methyl']
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
def str2num(seq):
    tmp_list = []
    for i in seq:
        tmp_list.append(label2id[i])
    return tmp_list

train_df['label_ids'] = list(map(lambda x: str2num(x), train_df['ner_tags']))


MAX_SEQ_LEN = 50

# define saving path
checkpoint_path = '../DR-DNA-model/checkpoint'
log_path = '../DR-DNA-model/log'
model_path = '../DR-DNA-model/model'
savingPath = [checkpoint_path, log_path, model_path]
for dir in savingPath:
    if not os.path.exists(dir):
        os.makedirs(dir)

# process dataset
tmp_train_dataset, tmp_test_dataset = train_test_split(train_df[['sentence', 'ner_tags', 'label_ids']], test_size=0.2, random_state=22)
tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name_or_path='wenhuan/MuLan-Methyl-DistilBERT_5hmC')

# add new tokens to the tokenizer vocabulary
new_tokens_3mer = set(list_3mer) - set(tokenizer.vocab.keys())
new_token_ = list(set(new_token+list(new_tokens_3mer)))
tokenizer.add_tokens(new_token_)
model = DistilBertCRF_Focal.from_pretrained(pretrained_model_name_or_path='wenhuan/MuLan-Methyl-DistilBERT_5hmC', num_labels=3, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))
print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# tokenizer

train_encodings = tokenizer(tmp_train_dataset['sentence'].tolist(), add_special_tokens=False, max_length = MAX_SEQ_LEN, padding = 'max_length', truncation = True, return_tensors = "pt")
test_encodings = tokenizer(tmp_test_dataset['sentence'].tolist(), add_special_tokens=False, max_length = MAX_SEQ_LEN, padding = 'max_length', truncation = True, return_tensors = "pt")


class NerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.LongTensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NerDataset(train_encodings, tmp_train_dataset['label_ids'].tolist())
test_dataset = NerDataset(test_encodings, tmp_test_dataset['label_ids'].tolist())
data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.flatten()
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro')
    return {
        'acc': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }


training_args = TrainingArguments(
    output_dir=checkpoint_path,          # output directory
    num_train_epochs=128,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=100,# number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir=log_path,            # directory for storing logs
    do_predict=True,
    learning_rate=3e-5,#3e-5
    disable_tqdm=False,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    do_eval=True,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    logging_strategy='epoch',
    save_total_limit=1,
    seed=42,
    report_to='tensorboard'
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args, # training arguments, defined above
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,             # evaluation dataset
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.evaluate(eval_dataset=test_dataset)

trainer.save_model(model_path)


