import pandas as pd
import torch



# seq2r
def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def seq2kmer_3mer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    kmers_s = f'N{kmers[0:2]}'
    kmers_e = f'{kmers[-2:]}N'
    kmers_expand = f'{kmers_s} {kmers} {kmers_e}'
    return kmers_expand


#train_set['sequence'] = list(map(lambda x: seq2kmer(x, 6), train_set['text']))

# assigned ner label
def label_ner(sequence, label):
    seq = sequence.split(' ')
    label_list = []
    if label == 1:
        for pos in range(len(seq)):
            if pos not in range(15, 21):
                label_list.append("O")
            else:
                label_list.append("I-LOC")
    else:
        for pos in range(len(seq)):
            label_list.append("O")
    return label_list


def label_ner_1mer(sequence, methy_list):
    seq = sequence.split(' ')
    label_list = ['O'] * len(seq)
    if len(methy_list) != 0:
        for pos in methy_list:
            label_list[pos] = 'Methyl'
    return label_list

def label_ner_1mer_3label(sequence, methy_list):
    seq = sequence.split(' ')
    indices = [i for i, x in enumerate(seq) if x == 'C']
    label_list = ['O'] * len(seq)
    if len(methy_list) != 0:
        nonMethyl_list = list(set(indices)-set(methy_list))
        for pos in methy_list:
            label_list[pos] = 'Methyl'
        for idx in nonMethyl_list:
            label_list[idx] = 'Non-Methyl'
    else:
        for pos in indices:
            label_list[pos] = 'Non-Methyl'


def label_ner_3mer_3label(sequence, methy_list):
    seq = sequence.split(' ')
    label_list = ['O'] * len(seq)
    centerC = []
    for i in range(len(seq)):
        if seq[i][1] == 'C':
            centerC.append(i)
    nonMethyl_list = list(set(centerC)-set(methy_list))
    if len(methy_list) != 0:
        for pos in methy_list:
            label_list[pos] = 'Methyl'
        for idx in nonMethyl_list:
            label_list[idx] = 'Non-Methyl'
    else:
        for pos in nonMethyl_list:
            label_list[pos] = 'Non-Methyl'
    return label_list

def label_ner_3mer_4label(sequence, methy_list):
    '''
    each token is a 3mer sequence
    four labels: 'O', 'B-Methyl', 'C-Methyl', 'E-Methyl'
    '''
    seq = sequence.split(' ')
    label_list = ['O'] * len(seq)
    if len(methy_list) != 0:
        for pos in methy_list:
            label_list[pos] = 'C-Methyl'
            try:
                label_list[pos-1] = 'E-Methyl'
            except:
                pass
            try:
                label_list[pos+1] = 'B-Methyl'
            except:
                pass
    return label_list

# list all possible 3 mer permutations
def kmer_permutation(list_):
    res = []
    for i in list_:
        for j in list_:
            for k in list_:
                ele = f'{i}{j}{k}'
                res.append(ele)
    return res


def func_CG(list_):
    list_ = list_.split(' ')
    new_list = []
    for ele in list_:
        if ele[1] == 'C':
            if 'CG' in ele:
                value = 1
            else:
                value = 2
        else:
            value = 0
        new_list.append(value)
    return new_list

# define dataset class
class MyDataset(Dataset):
    def __init__(self, in_dataset, tokenizer, label=True):
        self.data = in_dataset['sentence']
        self.tokenizer = tokenizer
        self.CG_anno = in_dataset['CG_anno']
        if label == True:
            self.labels = in_dataset['label_ids']
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]
        CG_anno = self.CG_anno[index]
        if self.labels != None:
            label = self.labels[index]

        # Tokenize the input text
        encoding = self.tokenizer(
            data_sample,
            add_special_tokens=False,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        # Return the input ids, attention mask, and label as tensors
        if self.labels != None:
            return input_ids, attention_mask, torch.tensor(CG_anno), torch.tensor(label)
        else:
            return input_ids, attention_mask, torch.tensor(CG_anno)