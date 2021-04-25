# coding: UTF-8
import os
import sys
BASEDIR = os.path.abspath(os.path.dirname(__file__))
print(BASEDIR)
sys.path.append(BASEDIR)
import time
import torch
import argparse
import numpy as np
from pprint import pprint
from importlib import import_module
from sklearn.metrics import classification_report
from train_eval import train, init_network, test
from utils import build_dataset, build_iterator, json_to_local


def parse():
    parser = argparse.ArgumentParser(description='Text Classification')
    # 选择模型
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='choose a model: Bert, ERNIE')
    # 数据目录
    parser.add_argument('--dataset',
                        type=str,
                        default='live_data',
                        # required=True,
                        help='dataset dir')
    # 网络参数初始化方法
    parser.add_argument('--init_method',
                        default='xavier',
                        type=str,
                        help='xavier or kaiming')
    parser.add_argument('--test',
                        type=bool,
                        default=False,
                        help='only test, defaut False')

    parser.add_argument('--seed', 
                        type=int, 
                        default=1, 
                        help='设置随机种子，default=1')

    parser.add_argument('--rate',
                        type=float,
                        default=0.1,
                        help='unsafe confidence interval, default=0.1')
    parser.add_argument('--testset',
                        type=str,
                        help='testset dir')
    parser.add_argument('--train_path',
                        type=str,
                        help='train path')
    parser.add_argument('--test_path',
                        type=str,
                        help='test path')
    args = parser.parse_args()
    return args


def set_rand_seed(seed=1):
    """设置随机种子"""
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False       
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  


if __name__ == '__main__':
    #dataset = 'THUCNews'  # 数据集
    #dataset = 'data'  # 数据集
    args = parse()
    set_rand_seed(args.seed)
    dataset = args.dataset  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    
    if args.train_path:
        config.train_path = args.train_path
    if args.test_path:
        config.test_path = args.test_path

    start_time = time.time()
    print("Loading data...")
    #print("config:", config)
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    model = x.Model(config).to(config.device)
    if args.test:
        test(config, model, test_iter, rate=args.rate)
    else:
        train(config, model, train_iter, dev_iter, test_iter)
