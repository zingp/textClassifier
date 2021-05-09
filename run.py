# coding: UTF-8
import os
import sys
BASEDIR = os.path.abspath(os.path.dirname(__file__))
print(BASEDIR)
sys.path.append(BASEDIR)
import time
import argparse
from pprint import pprint
from importlib import import_module
from train_eval import train, init_network, test
from utils import build_dataset, build_iterator, set_rand_seed


def parse():
    parser = argparse.ArgumentParser(description='Text Classification')
    # 选择模型
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='choose a model: bert, ERNIE')
    # 数据目录
    parser.add_argument('--dataset',
                        type=str,
                        default='data',
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
                        default=123, 
                        help='设置随机种子，default=123')
    parser.add_argument('--rate',
                        type=float,
                        default=0.5,
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


if __name__ == '__main__':
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
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    # train
    model = x.Model(config).to(config.device)
    if args.test:
        test(config, model, test_iter, rate=args.rate)
    else:
        train(config, model, train_iter, dev_iter, test_iter)
