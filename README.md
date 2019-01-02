# Use BERT as feature
# 环境
mac:
tf==1.4.0
python=2.7
windows:
tf==1.12
python=3.5
# 入口
调用预训练的模型，来做句子的预测。
bert_as_feature.py

配置data_root为模型的地址

调用预训练模型：chinese_L-12_H-768_A-12
# 最终结果
最后一层和倒数第二层：
last shape:(1, 14, 768), last2 shape: (1, 14, 768)
