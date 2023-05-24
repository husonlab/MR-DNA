import pandas as pd
import torch

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

# assign ner label
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
