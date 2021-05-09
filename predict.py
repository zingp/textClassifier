# coding: UTF-8
import os
import sys
BASEDIR = os.path.abspath(os.path.dirname(__file__))
print(BASEDIR)
sys.path.append(BASEDIR)

import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from datetime import timedelta
from train_eval import init_network, test
from importlib import import_module
from utils import  get_time_dif
from sklearn import metrics

random.seed(123)

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

def parser():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
    parser.add_argument('--rate', type=float, default=0.5, help='unsafe prob rate')
    args = parser.parse_args()
    return args


def listdir(basedir):
    files = []
    file_list = os.listdir(basedir)
    for fil in file_list:
        filename = os.path.join(basedir, fil)
        files.append(filename)
    return files


def split_sentence(s, a=80, b=150):
    start = 0
    n = len(s)
    reusult = []
    while start < n:
        gap = random.randint(80, 150)
        end = start + gap
        while end <= n:
            if s[end-1] == "，" or s[end-1] == "。":
                line = s[start: end].strip().rstrip("\n")
                reusult.append(line)
                break
            else:                
                end += 1
        else:
            line = s[start:].strip().rstrip("\n")
            if len(line) >= 40:
                reusult.append(line)
        start = end 
    return reusult


def build_tensor(s, config):
    pad_size = config.pad_size
    device = config.device
    contents = []
    sentence_list = split_sentence(s)
    for sentence in sentence_list:
        token = config.tokenizer.tokenize(sentence)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents.append((token_ids, -1, seq_len, mask))

    x = torch.LongTensor([_[0] for _ in contents]).to(device)
    y = torch.LongTensor([_[1] for _ in contents]).to(device)

    # pad前的长度(超过pad_size的设为pad_size)
    seq_len = torch.LongTensor([_[2] for _ in contents]).to(device)
    mask = torch.LongTensor([_[3] for _ in contents]).to(device)
    return (x, seq_len, mask), y


def load_data_from_file(filename):
    x_list = []
    y_list = []
    with open(filename, "r") as f:
        for line in f:
            if len(line.strip("\n").strip()) <= 50:
                continue
            x, y = line.strip("\n").split("\t")
            x_list.append(x)
            y_list.append(int(y))
    return x_list, y_list


def build_dataset(filename, config):
    x_tensor_list = []
    x_list, y_list = load_data_from_file(filename)
    for s in x_list:
        texts, _ =  build_tensor(s, config)
        x_tensor_list.append(texts)
    return x_tensor_list, y_list


def main_predict(model, test_file, config, rate=0.5):
    start_time = time.time()
    print("start build dataset...")
    x_list, y_list = build_dataset(test_file, config)
    print("build data done.")
    y_true = np.array(y_list, dtype=int)
    y_preds = []
    for idx, texts in enumerate(x_list):
        outputs = model(texts)
        pred_softmax = F.softmax(outputs.data, dim=1)
        # print("soft_max:",pred_softmax)
        pred = (pred_softmax[::, 1] >= rate)
        if torch.sum(pred) >= 1:
            y_preds.append(1)
        else:
            y_preds.append(0)
        if idx % 5000 == 0:
            end = time.time()
            print("idx: {}, duration: {}".format(idx, end-start_time))
            start_time = time.time()

    y_preds = np.array(y_preds, dtype=int)
    test_acc = metrics.accuracy_score(y_true, y_preds)
    report = metrics.classification_report(y_true, y_preds, target_names=config.class_list, digits=4)
    confusion = metrics.confusion_matrix(y_true, y_preds)
    
    print("Test Acc:", test_acc)
    print("Precision, Recall and F1-Score...")
    print(report)
    print("Confusion Matrix...")
    print(confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    

if __name__ == '__main__':
    dataset = 'live_data'              # 数据集
    basedir = "./live_data/test_data"  # 预测数据集

    args = parser()
    model_name = args.model  
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  

    start_time = time.time()
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    print("Params size {} M.".format(
        sum(n.numel() for n in model.parameters())*4/1024/1024
    ))
    model.eval()

    file_list = listdir(basedir)
    for test_file in file_list:
        print("*"*100)
        print("File: {}, rate: {}.".format(test_file, args.rate))
        main_predict(model, test_file, config, rate=args.rate)
