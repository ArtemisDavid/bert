# coding=utf-8
import pandas as pd
from sklearn import model_selection

data_path = "data/atec/all.csv"

all_data = pd.read_csv(data_path, sep='\t', names=['line', 'seq1', 'seq2', 'label'], header=-1)
print(all_data.shape)
print(all_data['seq1'].str.len().max())  # 97
print(all_data['seq2'].str.len().max())  # 112
print(all_data['seq2'].str.cat(all_data['seq1']).str.len().max())  # 166

x = all_data[['line', 'seq1', 'seq2']]
y = all_data['label']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=2019)

x_train['label'] = y_train
x_train.to_csv('data/atec/train.csv', index=False, sep='\t', encoding='utf-8', header=False)

x_test['label'] = y_test
x_test.to_csv('data/atec/test.csv', index=False, sep='\t', encoding='utf-8', header=False)


pass
