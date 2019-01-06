# coding=utf-8
import pandas as pd

data_path = "data/atec/all.csv"

all_data = pd.read_csv(data_path, sep='\t', names=['line', 'seq1', 'seq2', 'label'], header=-1)

print(all_data['seq1'].str.len().max())  # 97
print(all_data['seq2'].str.len().max())  # 112
print(all_data['seq2'].str.cat(all_data['seq2']).str.len().max())   # 224
