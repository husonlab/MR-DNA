'''
run fine-tuned DistilBERT+CRF model on the test set of custom database, each record of l00 length
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from transformers import DistilBertTokenizerFast, DistilBertTokenizer, DistilBertForMaskedLM, ElectraTokenizer, ElectraForMaskedLM, DistilBertConfig, ElectraConfig
from transformers import DistilBertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification, \
    AdamW, get_linear_schedule_with_warmup, XLNetForSequenceClassification, T5ForConditionalGeneration, AlbertForSequenceClassification, \
    ElectraForSequenceClassification, ElectraConfig, DistilBertForTokenClassification, DataCollatorForTokenClassification
import process_data
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments,AutoModelForSequenceClassification, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, matthews_corrcoef, classification_report, confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
import multiprocessing
from transformers import DataCollatorForLanguageModeling
from Bio.Seq import Seq
from datasets import load_metric
import evaluate
import numpy as np
#import process_data as process_data
#from test_distilbert_crf import DistilBertCRF, DistilBertCRF_Focal, DistilBertCRF_Focal_update
from ast import literal_eval
import custom_database.process_data as process_data
from custom_database.test_distilbert_crf import DistilBertCRF, DistilBertCRF_Focal, DistilBertCRF_Focal_update
import matplotlib.pyplot as plt
import seaborn as sns

length_list = ['100_sublength_50_gap', '200_sublength_100_gap','50_sublength_25_gap']
length_ = length_list[2]

test_set = pd.read_csv(f'/mnt/volume/project/5mC/db/seq_res/{length_}/test_sublength_seq_8hc_filtered_M.txt', sep='\t', converters={'methy_pos': literal_eval})
print(test_set.shape)
test_set = test_set.rename(columns={'sequence': 'text', 'methy_pos':'label'})
test_set = test_set[['text', 'label']]

# drop duplicate(not in paper)
test_set['new_label'] = test_set['label'].apply(lambda x: ', '.join(map(str, x)))
test_set = test_set.drop(['label'], axis=1)
test_set = test_set.drop_duplicates(keep='first')
test_set['label'] = test_set['new_label'].apply(lambda x: x.split(', '))
test_set['label'] = test_set['label'].apply(lambda x: [int(i) for i in x])
test_set = test_set.drop(['new_label'], axis=1) #31594
#test_set['sentence'] = list(map(lambda x: process_data.seq2kmer(x, 6), test_set['text']))
#test_set['ner_tags'] = list(map(lambda x, y: process_data.label_ner(x, y), test_set['sentence'], test_set['label']))
#test_set['sentence'] = list(map(lambda x: process_data.seq2kmer(x, 1), test_set['text']))
#test_set['ner_tags'] = list(map(lambda x, y: process_data.label_ner_1mer_3label(x, y), test_set['sentence'], test_set['label']))
test_set['sentence'] = list(map(lambda x: process_data.seq2kmer_3mer(x, 3), test_set['text']))
test_set['ner_tags'] = list(map(lambda x, y: process_data.label_ner_3mer_3label(x, y), test_set['sentence'], test_set['label']))

test_df = test_set[['ner_tags', 'sentence', 'label']]

# statistics new tokens (contains N)

test_set['sentence'] = list(map(lambda x: process_data.seq2kmer_3mer(x, 3), test_set['text']))
tmp_kmer_list = ' '.join(list(test_set['sentence']))
uniq_kmer_list = list(set(tmp_kmer_list.split(' ')))
new_token = []
for ele in uniq_kmer_list:
    if 'N' in ele:
        new_token.append(ele)


label_names = ['O', 'Methyl', 'Non-Methyl']
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
def str2num(seq):
    tmp_list = []
    for i in seq:
        tmp_list.append(label2id[i])
    return tmp_list

test_df['label_ids'] = list(map(lambda x: str2num(x), test_df['ner_tags']))

pretrainedLM = 'distilbert'
if pretrainedLM == 'electra':
    pretrained_model_name_or_path = 'google/electra-base-generator'
    my_tokenizer, config_file, masked_language_model, pretrained_model_for_sequence_classification = [ElectraTokenizer, ElectraConfig, ElectraForMaskedLM, ElectraForSequenceClassification]
elif pretrainedLM == 'distilbert':
    pretrained_model_name_or_path = 'distilbert-base-cased'
    my_tokenizer, config_file, masked_language_model, pretrained_model_for_token_classification = [DistilBertTokenizerFast, DistilBertConfig, DistilBertForMaskedLM, DistilBertForTokenClassification]

checkpoint_path = f'{pretrainedLM}/mlm/checkpoint'
log_path = f'{pretrainedLM}/mlm/log'
mlm_model_path = f'{pretrainedLM}/mlm/model'

length_map = {'100_sublength_50_gap': ['NER_CRF_100_3mer_3label_alpha_181', 100],
              '200_sublength_100_gap': ['NER_CRF_200_3mer_3label_alpha', 200],
              '50_sublength_25_gap': ['NER_CRF_3mer_3label_alpha_172', 50]}
model_subpath, MAX_SEQ_LEN = length_map[length_]

finetune_checkpoint_path = f'{pretrainedLM}/finetune/5mC_customLength/{model_subpath}/checkpoint'
finetune_log_path = f'{pretrainedLM}/finetune/5mC_customLength/{model_subpath}/log'
finetune_model_path = f'{pretrainedLM}/finetune/5mC_customLength/{model_subpath}/model'
#finetune_model_path = f'{pretrainedLM}/finetune/5mC_customLength/{model_subpath}/checkpoint/checkpoint-8492'

tokenizer = my_tokenizer.from_pretrained(f'/mnt/volume/project/5mC/pretrainedModel/{pretrainedLM}/tokenizer/{pretrainedLM}_seq_tax_trained', num_labels=3)
# add new tokens to the tokenizer vocabulary
tokenizer.add_tokens(new_token)

class NerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.LongTensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

x_test = test_df['sentence'].tolist()
test_label = test_df['label_ids'].tolist()

test_encodings = tokenizer(x_test, add_special_tokens=False, max_length = MAX_SEQ_LEN, padding = 'max_length', truncation = True, return_tensors = "pt")

testDataset = NerDataset(test_encodings, test_label)
data_collator = DataCollatorForTokenClassification(tokenizer)

model = DistilBertCRF_Focal.from_pretrained(f'/mnt/volume/project/5mC/pretrainedModel/{finetune_model_path}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
'''
def compute_metrics(pred):
    #labels = pred.label_ids[:, 1:-1]
    labels = pred.label_ids.flatten()
    preds = pred.predictions.flatten()
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    precision = precision_score(labels, preds)
    #print(classification_report(labels, preds))
    return {
        'acc': acc,
        'f1': f1,
        'recall': recall,
        'mcc': mcc,
        'precision': precision
    }
'''

def compute_metrics(pred):
    #labels = pred.label_ids[:, 1:-1]
    labels = pred.label_ids.flatten()
    preds = pred.predictions.flatten()
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='macro')
    #mcc = matthews_corrcoef(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))
    cm = confusion_matrix(labels, preds)
    #ConfusionMatrixDisplay(cm).plot()
    return {
        'acc': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }

args = TrainingArguments(output_dir='tmp_trainer', per_device_eval_batch_size=32, do_predict=True)
trainer = Trainer(args=args, model=model, tokenizer=tokenizer, data_collator=data_collator)
pred = trainer.predict(testDataset)
print(len(pred.predictions))

result = compute_metrics(pred)

print(result)


# confusion matrix visualization

cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2%', xticklabels=['O', 'Methyl', 'Non-Methyl'], yticklabels=['O', 'Methyl', 'Non-Methyl'])
plt.savefig('/mnt/volume/project/5mC/figure/cm_200.pdf')
plt.show()


'''
from transformers import pipeline
token_classifier = pipeline(
    "token-classification", model=f'/home/ubuntu/project/dna_methy/pretrainedModel/{finetune_model_path}', tokenizer=tokenizer)
token_classifier = pipeline(
    "token-classification", model=f'/home/ubuntu/project/dna_methy/pretrainedModel/{finetune_model_path}', tokenizer=tokenizer)

token_classifier(test_df.iloc[1,1])
'''


'''
# beta version for predicting on the test dataset
def compute_metrics(label, pred):
    #labels = pred.label_ids[:, 1:-1]
    labels = torch.as_tensor(label).flatten()
    preds = pred.flatten().cpu()
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='macro')
    #mcc = matthews_corrcoef(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))
    return {
        'acc': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision
    }

model.eval()
inputs = testDataset.encodings.to(device)
with torch.no_grad():
    outputs = model(**inputs)

compute_metrics(testDataset.labels, outputs[1])
'''