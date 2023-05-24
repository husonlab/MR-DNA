import os
import sys
print(os.path.abspath(os.path.dirname(__file__)))
print(os.getcwd())
import pandas as pd
from ast import literal_eval
dataPath = '../database/MR-DNA-50/train.txt'
train_set = pd.read_csv(dataPath, sep='\t', converters={'methy_pos': literal_eval})
print(train_set)