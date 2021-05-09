# coding: UTF-8
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/train.txt'                                # 训练集
        self.dev_path = dataset + '/dev.txt'                                    # 验证集
        self.test_path = dataset + '/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/classes.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        if not os.path.isdir(dataset + '/saved_dict/'):
           os.makedirs(dataset + '/saved_dict/')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 2000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        #self.num_classes = 1                                           # 类别数
        self.num_epochs = 5                                             # epoch数
        self.batch_size = 16                                           # mini-batch大小
        self.max_len = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5   #5e-5                                       # 学习率
        self.bert_path = './prt_models/robert_wwm_ext/'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, mask):
       _, pooled = self.bert(input_ids, attention_mask=mask)
       out = self.fc(pooled)
       return out
