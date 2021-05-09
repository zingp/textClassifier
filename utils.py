# coding: UTF-8
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from datetime import timedelta

def timer(module):
    """统计函数耗时的装饰器
    Args:
        module (str): Description of the function being timed.
    """
    def wrapper(func):
        def cal_time(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            duration = round(t2 - t1, 4)
            print('{} duration={} secs.'.format(module, duration))
            return res
        return cal_time
    return wrapper


def build_dataset(config):
    @timer("load_dataset")
    def load_dataset(path, max_len=128):
        tokernizer = config.tokenizer
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                line = line.strip()
                # 如果是空行
                if not line:
                    continue
                text, label = line.split('\t')
                inputs_dict = tokernizer.encode_plus(text,
                                            padding='max_length',
                                            truncation=True,
                                            max_length=max_len,  
                                            pad_to_max_length=True, 
                                            return_attention_mask=True,
                                            return_tensors="pt")
                input_ids = inputs_dict['input_ids']
                attention_mask = inputs_dict['attention_mask']
                contents.append((input_ids, attention_mask, int(label)))
        return contents

    train = load_dataset(config.train_path, config.max_len)
    dev = load_dataset(config.dev_path, config.max_len)
    test = load_dataset(config.test_path, config.max_len)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if self.n_batches and len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.cat([_[0] for _ in datas], 0).to(self.device)
        mask = torch.cat([_[1]  for _ in datas], 0).to(self.device)
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return x, mask, y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def set_rand_seed(seed=123):
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


def time_diff(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

