# Use BERT as feature
# 环境
```python
mac:
tf==1.4.0
python=2.7

windows:
tf==1.12
python=3.5
```

# 入口
调用预训练的模型，来做句子的预测。
bert_as_feature.py

配置data_root为模型的地址

调用预训练模型：chinese_L-12_H-768_A-12
# 最终结果
最后一层和倒数第二层：
last shape:(1, 14, 768), last2 shape: (1, 14, 768)

```
# last value
[[ 0.8200665   1.7532703  -0.3771637  ... -0.63692784 -0.17133102
   0.01075665]
 [ 0.79148203 -0.08384223 -0.51832616 ...  0.8080162   1.9931345
   1.072408  ]
 [-0.02546642  2.2759912  -0.6004753  ... -0.88577884  3.1459959
  -0.03815675]
 ...
 [-0.15581022  1.154014   -0.96733016 ... -0.47922543  0.51068854
   0.29749477]
 [ 0.38253042  0.09779643 -0.39919692 ...  0.98277044  0.6780443
  -0.52883977]
 [ 0.20359193 -0.42314947  0.51891303 ... -0.23625426  0.666618
   0.30184716]]
```



# 预处理

`tokenization.py`是对输入的句子处理，包含两个主要类：`BasickTokenizer`, `FullTokenizer`

`BasickTokenizer`会对每个字做分割，会识别英文单词，对于数字会合并，例如：

```
query: 'Jack,请回答1988, UNwant\u00E9d,running'
token: ['jack', ',', '请', '回', '答', '1988', ',', 'unwanted', ',', 'running']
```

`FullTokenizer`会对英文字符做n-gram匹配，会将英文单词拆分，例如running会拆分为run、##ing，主要是针对英文。

```
query: 'UNwant\u00E9d,running'
token: ["un", "##want", "##ed", ",", "runn", "##ing"]
```

对于中文数据，特别是NER，如果数字和英文单词是整体的话，会出现大量UNK，所以要将其拆开，想要的结果：

```
query: 'Jack,请回答1988'
token:  ['j', 'a', 'c', 'k', ',', '请', '回', '答', '1', '9', '8', '8']
```

具体变动如下：

```python
class CharTokenizer(object):
    """Runs end-to-end tokenziation."""
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in token:
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)
```







