# coding: UTF-8
import os
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        if not os.path.isdir(dataset + '/saved_dict/'):
           os.makedirs(dataset + '/saved_dict/')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        #self.num_classes = 1                        # 类别数
        self.num_epochs = 5                                             # epoch数
        self.batch_size = 16                                           # mini-batch大小
        self.pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5   #5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        """
        参数:
        	config: bert模型的参数
    		输入和BertForSequenceClassification基本一样，少了标签，多了一个output_all_encoded_layers用来控制是否输出每一层的状态   
    		输出是shape为 (encoded_layers, pooled_output)的元组
        	`encoded_layers`: 通过output_all_encoded_layers参数控制:
                `output_all_encoded_layers=True`: 输出每个层encoded-hidden-states的列表，每层状态的类型是FloatTensor，shape是 [batch_size, sequence_length, hidden_size]
                `output_all_encoded_layers=False`: 只输出最后一层状态，shape是[batch_size, sequence_length, hidden_size],
                `pooled_output`: 用来训练两句话类型的bert模型的输出，FloatTensor类型，shape是[batch_size, hidden_size] 
           https://zhuanlan.zhihu.com/p/56155191
       """
        out = self.fc(pooled)
        return out
