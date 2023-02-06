import jsonlines
from pymysql.converters import escape_string
import argparse
import json
import pymysql
import traceback
import os
from dbutils.pooled_db import PooledDB
from prettyprinter import cpprint
import logging
from prettyprinter import cpprint, pformat
from concurrent_log_handler import ConcurrentRotatingFileHandler
from omegaconf import OmegaConf
import torch
import sys
import requests
import time
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LOG_MAX_BYTES = 1024 * 1024 * 10
LOG_PATH_LIST = None
LOG_BASE_PATH = "log"
print_raw = print
#    这条注释不能删除，依靠这个井号键来分割
import jsonlines
import torch
import hydra
from prettyprinter import cpprint, pformat
from collections import Counter
import json
from hydra.utils import get_original_cwd, to_absolute_path
from transformers import BertTokenizerFast
import shutil
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from torch.utils.data import DataLoader
from alive_progress import alive_bar
from abc import ABC, abstractmethod
from prettyprinter import cpprint
import os
from PIL import Image
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from torch import Tensor, nn
import omegaconf
from bert4torch.models import *
from transformers import LayoutXLMProcessor
import traceback
from sklearn.metrics import classification_report
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict, Counter
from asyncio.log import logger
from transformers import BertTokenizer
from multiprocessing import reduction
from torch import Tensor
from os import mkdir
import base64
from collections import defaultdict
from genericpath import exists
import copy
import math
from torch.nn import LogSoftmax
import regex as re
import pandas as pd
import random
from torch import nn
import inspect
from omegaconf import DictConfig, OmegaConf
from abc import abstractmethod
import transformers
import requests
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch.nn.functional as F
REGISTED_MODEL = {}
METHOD_NAME_LIST = []
PACKAGE_LIST = []
GB = {}


def register_model(name):
    def register_module_cls(cls):
        REGISTED_MODEL[name] = cls
        return cls

    return register_module_cls


@register_model('ioperation')
def ioperation(fn_name,
               fn_inputs_kws={},
               scope_super={},
               module_dict={},
               class_attr=None):
    """
    比较重要的方法，所有REGIATED_MODEL的调用都要通过此方法，

    1. 会做用到方法的记录。代码生成会用到

    Args:
        fn_name (_type_): _description_
        fn_inputs_kws (dict, optional): _description_. Defaults to {}.
        scope_super (dict, optional): _description_. Defaults to {}.
        module_dict (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    REGISTED_MODEL_LOWER_MAP = {}
    for k, v in REGISTED_MODEL.items():
        REGISTED_MODEL_LOWER_MAP[k.lower()] = k
    scope_local = locals()
    _kwargs = {}
    for k, v in fn_inputs_kws.items():
        if isinstance(v, str) and re.findall('(^var_|global_)', v):
            _var = re.sub('^var_', '', v)
            _kwargs.update({k: eval(_var, scope_super, scope_local)})
        else:
            _kwargs.update({k: v})

    if class_attr:
        METHOD_NAME_LIST.append(fn_name)
        REGISTED_MODEL['read_package'](fn_name)

        return getattr(
            REGISTED_MODEL[REGISTED_MODEL_LOWER_MAP[fn_name.lower()]],
            class_attr)(**_kwargs)
    else:
        if fn_name in module_dict:
            return module_dict[fn_name](**_kwargs)
        else:
            METHOD_NAME_LIST.append(fn_name)
            REGISTED_MODEL['read_package'](fn_name)
            return REGISTED_MODEL[REGISTED_MODEL_LOWER_MAP[fn_name.lower()]](
                **_kwargs)


@register_model('read_package')
def read_package(fn_name):
    REGISTED_MODEL_LOWER_MAP = {}
    for k, v in REGISTED_MODEL.items():
        REGISTED_MODEL_LOWER_MAP[k.lower()] = k
    with open(
            str(
                inspect.getfile(REGISTED_MODEL[REGISTED_MODEL_LOWER_MAP[
                    fn_name.lower()]])), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if re.findall('^import ', line) or re.findall(
                    '^from .* import ', line):
                if line not in PACKAGE_LIST:
                    PACKAGE_LIST.append(line)


@register_model('DataBase')
class DataBase(object):
    """
    databuffer和datasink以及sample的基类
    两个用处：
    1. 初始化之后不允许添加新的属性，保证所有人看到类之后就能知道有什么属性
    2. 任何属性第一次赋值之后不允许重复赋值

    保证这两点，可以大大减少数据在复杂处理过程中被意外更改添加造成的错误，特别是数据更改（造成训练结果差，但是没有bug），

    Args:
        object (_type_): _description_

    Raises:
        AttributeError: _description_
    """
    __freeze = False

    def __setattr__(self, key, value):
        if self.__freeze and not hasattr(self, key):
            raise AttributeError(f"不允许在databuffer中添加新的属性，会造成难以维护的问题")

        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__freeze = True

    def _unfreeze(self):
        self.__freeze = False


@register_model('SampleObj')
class SampleObj(object):

    __freeze = False

    def __setattr__(self, key, value):
        if self.__freeze and not hasattr(self, key):
            raise AttributeError(f"不允许在databuffer中添加新的属性，会造成难以维护的问题:{key}")
        assert self.__dict__.get(
            key) is None, f"{key} :{self.__dict__.get(key)} 不允许重复赋值"

        # self.__dict__[key] = value
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__freeze = True

    def _unfreeze(self):
        self.__freeze = False


@register_model('DataBase')
class DataBase(object):
    """
    databuffer和datasink以及sample的基类
    两个用处：
    1. 初始化之后不允许添加新的属性，保证所有人看到类之后就能知道有什么属性
    2. 任何属性第一次赋值之后不允许重复赋值

    保证这两点，可以大大减少数据在复杂处理过程中被意外更改添加造成的错误，特别是数据更改（造成训练结果差，但是没有bug），

    Args:
        object (_type_): _description_

    Raises:
        AttributeError: _description_
    """
    __freeze = False

    def __setattr__(self, key, value):
        if self.__freeze and not hasattr(self, key):
            raise AttributeError(f"不允许在databuffer中添加新的属性，会造成难以维护的问题")

        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__freeze = True

    def _unfreeze(self):
        self.__freeze = False


@register_model('collate_bert_cla')
def collate_bert_cla(batch):
    all_input_ids = np.array(
        [x['sample'].get_features()['token_ids'] for x in batch])
    all_attention_mask = np.array(
        [x['sample'].get_features()['masks'] for x in batch])
    all_segment_ids = np.array(
        [x['sample'].get_features()['segment_ids'] for x in batch])
    all_lens = np.array([x['sample'].get_features()['lens'] for x in batch])

    all_labels = [x['sample'].get_features()['label_ids'] for x in batch]

    all_labels = None if not any(all_labels) else all_labels
    max_len = max(all_lens)

    # 现在label是   [[0],[0],[1],[1]]  ,要取出来
    if all_labels is None:
        all_labels = None
    elif all([len(x) == 1 for x in all_labels]):
        all_labels = [x[0] for x in all_labels]
        all_labels = np.array(all_labels)
    else:
        all_labels = np.array(all_labels)
        # TODO 分类任务对label不需要操作
        # all_labels=all_labels[:,:max_len]

    all_idx = [x['idx'] for x in batch]
    all_input_ids = torch.tensor(all_input_ids[:, :max_len], dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask[:, :max_len],
                                      dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids[:, :max_len],
                                   dtype=torch.long)
    all_labels = torch.tensor(
        all_labels, dtype=torch.long) if all_labels is not None else None
    all_lens = torch.tensor(all_lens, dtype=torch.long)

    if all_labels is not None:
        return {
            "inputs":
            (all_input_ids, all_attention_mask, all_segment_ids, all_labels),
            "idx":
            all_idx
        }
    else:
        return {
            "inputs": (all_input_ids, all_attention_mask, all_segment_ids),
            "idx": all_idx
        }


@register_model('op_inplace')
def op_inplace(input):
    return input


@register_model('pipline_validation_epoch')
def pipline_validation_epoch():  # 相同的数据，多跑几轮``
    with torch.no_grad():
        datasink = GB['databuffer'].get_datasink('validation')
        with alive_bar(
                len(datasink.dataloader),
                title=f"pipline_validation_epoch:{GB['databuffer'].epoch_count}"
        ) as bar:
            for datasink.batch_inputs in datasink.dataloader:
                bar()
                GB['databuffer'].call_hooks('validation', 'before_batch')
                if isinstance(datasink.batch_inputs['inputs'], dict):
                    GB['databuffer'].model.forward_prediction(
                        **{
                            k: v.to(GB['databuffer'].device)
                            for k, v in
                            datasink.batch_inputs['inputs'].items()
                        })
                else:
                    GB['databuffer'].model.forward_prediction(
                        tuple(
                            t.to(GB['databuffer'].device)
                            for t in datasink.batch_inputs['inputs']))
                GB['databuffer'].call_hooks('validation', 'after_batch')


@register_model('op_argmax')
def op_argmax(logist, dim):
    logist = logist.max(dim=dim)
    logist = logist.indices.cpu().numpy().tolist()
    logist = [[x] for x in logist]
    return logist


@register_model('MarkdownBase')
class MarkdownBase():
    def __init__(self, base64=False) -> None:
        self.fig_dir = 'report/markdown/figure'
        os.makedirs(self.fig_dir, exist_ok=True)

        self.head = []
        self.content = []
        self.tail = []

        self.base64 = base64

    def write(self, file_name='report/markdown/实验报告.md'):
        with open(file_name, 'a') as f:
            f.write('  \n'.join(self.head))
            f.write('  \n'.join(self.content))
            f.write('  \n'.join(self.tail))

    def fill_form(self,
                  columns=['A', 'B', "C"],
                  data=[[1, 1, 1], [2, 2, 2], [3, 3, 3]]):

        assert len(columns) == len(next(iter(data)))

        self.content.append(' | ' + ' | '.join(columns) + ' | ')
        self.content.append(' | ' + ' | '.join(len(columns) * [' :----: ']) +
                            ' | ')
        for row in data:
            self.content.append(' | ' + ' | '.join(row) + ' | ')

    def draw_chart_horizontal_bar(self,
                                  title='draw_chart_horizontal_bar',
                                  x_label='f1_score',
                                  data={
                                      "A": 11,
                                      "B": 3,
                                      "C": 9
                                  }):
        plt.rcdefaults()

        labels = list(data.keys())
        performance = list(data.values())

        figure_path = os.path.join(self.fig_dir, title + '.png')
        labels, performance = (list(x) for x in zip(
            *sorted(zip(labels, performance), key=lambda x: x[1])))

        # plt.figure(dpi=300,figsize=(2,1) )
        fig, ax = plt.subplots()
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, performance, align='center')
        ax.set_yticks(y_pos, labels=labels)
        ax.invert_yaxis()
        ax.set_xlabel(x_label)
        ax.set_title(title)
        plt.savefig(figure_path)
        plt.show()
        plt.close()
        self.apply_figure(title, figure_path)

    def draw_chart_line(self,
                        title='loss_trends',
                        x_label='epoch',
                        y_label='step',
                        x_data=[0.5, 0.4, 0.3, 0.2, 0.1],
                        y_data={'loss': [1, 2, 3, 4, 5]}):
        plt.rcdefaults()
        figure_path = os.path.join(self.fig_dir, title + '.png')

        fig, ax = plt.subplots()

        # cpprint(x_data)
        # cpprint(y_data)
        for legend, data in y_data.items():
            ax.plot(x_data, data)
        ax.legend(list(y_data.keys()))
        ax.set(xlabel=x_label, ylabel=y_label, title=title)

        ax.grid()
        fig.savefig(figure_path)
        plt.show()

        plt.close()
        self.apply_figure(title, figure_path)

    def write_code(self, lang, code):

        self.content.append(f'```{lang}')
        self.content.append(code)
        self.content.append('```')

    def write_heading(self, content, level):

        self.content.append(f'{"#"*level} {content}')

    def _convert2base64(self, path):
        return base64.b64encode(open(path, 'rb').read()).decode('ascii')

    def write_content(self, content):

        self.content.append(content)

    def apply_figure(self, title, figure_path):
        if self.base64 is True:
            self.content.append(f'![{title}][{title}]')

            self.tail.append(
                f'[{title}]data:image/png;base64,{self._convert2base64(figure_path)}'
            )

        else:
            self.content.append(f'![{title}](figure/{title}.png)')


@register_model('NetBertBase')
class NetBertBase(transformers.models.bert.BertModel):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_config(cls, config):
        return cls(transformers.BertConfig.from_dict(config))


@register_model('create_label')
def create_label(**args):
    """
    将 ner 的truth转化为对应的label

    # 改成processor的 truth2label
    """
    datasink = GB['databuffer'].get_datasink(GB['databuffer'].procedure)
    for sample in datasink.get_sample():
        if sample.sample_truth is not None:
            sample.sample_label = GB['databuffer'].processor.truth2label(
                sample.sample_text, sample.sample_truth)
            sample.orig_label = GB['databuffer'].processor.truth2label(
                sample.orig_text, sample.orig_truth)


@register_model('gen_requirements')
def gen_requirements():

    return list(set(PACKAGE_LIST))


@register_model('ArchBase')
class ArchBase(nn.Module):
    def __init__(self, **args) -> None:
        """
        args['model_config']['model_arch']  ：  模型结构

        args['model_config']['forward_logist']   forward的流程

        args['model_config']['forward_training']   训练的流程
        

        args['model_config']['forward_prediction']   预测的流程



        1. 其中forward是 训练和预测的共通的部分，提取出来放在forward里面
        2. training 的结果是两个loss，一个是batch的总体loss，一个batch中每个样例的loss
        3. 没有返回值，返回保存在对应的datasink的  batch_loss，batch_losses以及batch_pred_ids中
        """
        super().__init__()
        self.initializer_range = args['initializer_range']
        self.module_dict = nn.ModuleDict({})
        self.cfg_model = args['model_config']['model_arch']
        self.cfg_logist = args['model_config']['forward_logist']
        self.cfg_training = args['model_config']['forward_training']
        self.cfg_prediction = args['model_config']['forward_prediction']

        REGISTED_MODEL_LOWER_MAP = {}
        for k, v in REGISTED_MODEL.items():
            REGISTED_MODEL_LOWER_MAP[k.lower()] = k
        for module_info, module_inputs_kws in self.cfg_model.items():
            module_out, module_name = module_info.split('-')
            METHOD_NAME_LIST.append(module_name)
            REGISTED_MODEL['read_package'](module_name)
            self.module_dict.update({
                module_out:
                REGISTED_MODEL[REGISTED_MODEL_LOWER_MAP[
                    module_name.lower()]].from_config(module_inputs_kws)
            })
        self.init_weights()

    def _logist(self, args):
        scope_local = locals()
        for fn_info, fn_inputs_kws in self.cfg_logist.items():
            module_out, module_name = fn_info.split('-')
            scope_local[module_out] = REGISTED_MODEL['ioperation'](
                module_name, fn_inputs_kws, scope_local, self.module_dict)
        return scope_local['logist']

    def forward_training(self, args):
        # datasink=self.GB['databuffer'].get_datasink(self.GB['databuffer'].procedure)
        scope_local = locals()
        scope_local['logist'] = self._logist(args)
        for fn_info, fn_inputs_kws in self.cfg_training.items():
            module_out, module_name = fn_info.split('-')
            scope_local[module_out] = REGISTED_MODEL['ioperation'](
                module_name, fn_inputs_kws, scope_local, self.module_dict)

        # loss 不返回，而是放在datasink里面
        GB['databuffer'].get_datasink().batch_loss = scope_local['loss']
        GB['databuffer'].get_datasink().batch_losses = scope_local['losses']

    def forward_prediction(self, args):
        # datasink=self.GB['databuffer'].get_datasink(self.GB['databuffer'].procedure)
        scope_local = locals()
        scope_local['logist'] = self._logist(args)
        for fn_info, fn_inputs_kws in self.cfg_prediction.items():
            module_out, module_name = fn_info.split('-')
            scope_local[module_out] = REGISTED_MODEL['ioperation'](
                module_name, fn_inputs_kws, scope_local, self.module_dict)

        # loss 不返回，而是放在datasink里面
        GB['databuffer'].get_datasink().batch_pred_ids = scope_local['pred']
        # 返回结果 在onnx导出时使用
        return scope_local['logist']
        # return scope_local['pred']

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        self.apply(self._init_weights)


@register_model('loading_param_base')
def param_base_loader(strict=False, **args):
    """
    载入模型

    Args:
        strict (bool, optional): _description_. Defaults to False.
    """
    GB['databuffer'].model.load_state_dict(torch.load(args['loading_path'], map_location=GB['databuffer'].device),
                                           strict)
    _ = f'载入参数 from {args["loading_path"]}'
    print(_), logger.info(_)


@register_model('metric_microf1_base')
def metric_microf1_base():
    def compute(origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision *
                                                 recall) / (precision + recall)
        return recall, precision, f1

    datasink = GB['databuffer'].get_datasink(GB['databuffer'].procedure)

    #  计算sample的单个指标
    for idx, sample in enumerate(datasink.get_sample()):
        if sample.sample_counter is not None:
            class_info = {}
            truth_counter = sample.sample_counter['truth_counter']
            found_counter = sample.sample_counter['found_counter']
            right_counter = sample.sample_counter['right_counter']
            for type_, count in truth_counter.items():
                origin = count
                found = found_counter.get(type_, 0)
                right = right_counter.get(type_, 0)
                recall, precision, f1 = compute(origin, found, right)
                class_info[type_] = {
                    "precision": round(precision, 4),
                    'recall': round(recall, 4),
                    'f1': round(f1, 4)
                }
            truth = sum(dict(truth_counter).values())
            found = sum(dict(found_counter).values())
            right = sum(dict(right_counter).values())
            sample_recall, sample_precision, sample_f1 = compute(
                truth, found, right)
            sample.sample_score = {
                'precision': sample_precision,
                'recall': sample_recall,
                'f1': sample_f1,
                "detail": class_info
            }
        else:
            sample.sample_score = {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                "detail": 0
            }

    datasink = GB['databuffer'].get_datasink(GB['databuffer'].procedure)

    #  计算orig的单个指标，当 truncation不划分时，跟sample的结果是一样的
    truth_counter, found_counter, right_counter = Counter(), Counter(
    ), Counter()
    start = 0
    for t_len in datasink.truncation_step:
        for sample in datasink.get_sample()[start:start + t_len]:
            # print(sample.orig_counter['truth_counter'])
            truth_counter = truth_counter + sample.orig_counter['truth_counter']
            found_counter = found_counter + sample.orig_counter['found_counter']
            right_counter = right_counter + sample.orig_counter['right_counter']
            # 每组的counter是一样的，只需要累加一任意一个就行
            break
        start += t_len

    _ = f'========counter_result预览========'
    print(_), logger.info(_)
    _ = '========================================'
    print(_), logger.info(_)
    _ = f' truth_counter   {truth_counter}'
    print(_), logger.info(_)
    _ = f' found_counter   {found_counter}'
    print(_), logger.info(_)
    _ = f' right_counter   {right_counter}'
    print(_), logger.info(_)

    class_info = {}
    for type_, count in truth_counter.items():
        truth = count
        found = found_counter.get(type_, 0)
        right = right_counter.get(type_, 0)
        recall, precision, f1 = compute(truth, found, right)
        class_info[type_] = {
            "precision": round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
    truth = sum(dict(truth_counter).values())
    found = sum(dict(found_counter).values())
    right = sum(dict(right_counter).values())
    recall, precision, f1 = compute(truth, found, right)

    _ = f'======== Overall evaluation  ========'
    logger.info(_)
    # print(_), logger.info(_)
    _ = f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}"
    logger.info(_)
    # print(_), logger.info(_)
    _ = "======== Detail evaluation  ========"
    logger.info(_)
    # print(_), logger.info(_)
    for _type, _info in sorted(class_info.items()):
        _ = f"{_type}=====" + "-".join(
            [f'{key}: {value:.4f} ' for key, value in _info.items()])
        logger.info(_)
        # print(_), logger.info(_)

    logger.info('======== Overall evaluation  ========')
    logger.info(
        f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    logger.info("======== Detail evaluation  ========")
    for _type, _info in sorted(class_info.items()):
        _ = f"{_type}=====" + "-".join(
            [f'{key}: {value:.4f} ' for key, value in _info.items()])
        logger.info(_)
    datasink.history_epoch_score.append({
        'precision': precision,
        'recall': recall,
        'f1': f1,
        "detail": class_info
    })


@register_model('DatasetBase')
class DatasetBase(Dataset):
    """
    目的是为了在取数据的时候带着idx，这样可以定位到datasink里面的sample具体是哪一个

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            # sample 用来取里面的feature
            'sample': self.samples[idx],
            'idx': idx
        }


@register_model('show_examples_feature')
def show_examples_feature(show_members=None):
    def show(member):
        # 打印属性，目前只支持
        #  int,float,bool,complex
        #  或者用 tuple或者list把上面的类型包起来的数据
        member_data = sample.get_features()[member]
        if member_data is not None:
            if isinstance(member_data, (int, float, bool, complex)):
                _ = f"{member}: %s", str(member_data)
                logger.info(_)
                # print(_), logger.info(_)
            elif isinstance(member_data, (list, tuple)):
                if all([
                        isinstance(x, (int, float, bool, complex))
                        for x in member_data
                ]):
                    _ = f"{member}: %s", " ".join(
                        [str(x) for x in sample.get_features()[member]])
                    logger.info(_)
                    # print(_), logger.info(_)
            else:
                _ = f"show example_failed unsupport dtype: {member}"
                logger.info(_)
                # print(_), logger.info(_)

    datasink = GB['databuffer'].get_datasink()
    _ = "*** Example ***"
    logger.info(_)
    # print(_), logger.info(_)
    for (ex_index, sample) in enumerate(datasink.get_sample()[:2]):
        if show_members:
            # 如果指定要打印的属性，则打印对应属性，否则打印全部属性
            for member in show_members:
                show(member)
        else:
            show_members = list(
                sample.get_features().keys()
            )  #[attr for attr in dir(sample) if not callable(getattr(sample, attr)) and not attr.startswith("__") and not attr.startswith("_")]
            for member in show_members:
                show(member)


@register_model('gen_funcation_code')
def gen_funcation_code():
    # 先加入registed_model
    code = []
    pattern_clasname = '^class\s(?P<class_name>.*?)[\(:]'
    pattern_classparent = '\((.*)\)'
    global METHOD_NAME_LIST
    METHOD_NAME_LIST = list(set(METHOD_NAME_LIST))
    # print('\n'.join(list(set(method_name_list))))
    parent_code = []
    for fn_name in METHOD_NAME_LIST:
        source_code = inspect.getsource(REGISTED_MODEL[fn_name])
        if source_code.startswith('class '):
            search_obj = re.search(pattern_clasname, source_code)
            if search_obj:
                class_name = search_obj.group('class_name')
                source_code = f"@register_model('{class_name}')\n" + source_code
                class_name = source_code.split('\n')[1].replace("'", "")
                class_name = class_name.replace('"', "")
                class_name = class_name.replace("]", "")
                class_name = class_name.replace("REGISTED_MODEL[", "")

                pattern_classparent = '\((?P<parent>.*)\)'
                search_obj = re.search(pattern_classparent, class_name)
                if search_obj:
                    class_name = search_obj.group('parent')
                    # print(class_name)
                    if class_name in REGISTED_MODEL:
                        # tbd 多继承没写，以后补上
                        #  多层集成也没写
                        parent_code.append('\n')
                        parent_code.append(
                            f"@register_model('{class_name}')\n")
                        parent_code.append(
                            inspect.getsource(REGISTED_MODEL[class_name]))
                        parent_code.append('\n')
        code.append(source_code)
    code = parent_code + code
    # 再加入ioperation
    code = [inspect.getsource(REGISTED_MODEL['read_package'])] + code
    code = [inspect.getsource(REGISTED_MODEL['ioperation'])] + code
    code = [inspect.getsource(register_model)] + code

    code = [
        'REGISTED_MODEL={}', 'METHOD_NAME_LIST=[]', 'PACKAGE_LIST=[]', 'GB={}'
    ] + code
    return code


@register_model('activate_softmax')
def activate_softmax(input: Tensor,
                     dim: int = None,
                     _stacklevel: int = 3,
                     dtype: torch.dtype = None) -> Tensor:
    """
    激活函数: softmax,torch.nn.functional.softmax

    Args:
        input (Tensor): _description_
        dim (int, optional): _description_. Defaults to None.
        _stacklevel (int, optional): _description_. Defaults to 3.
        dtype (torch.dtype, optional): _description_. Defaults to None.

    Returns:
        Tensor: _description_
    """
    return F.softmax(input, dim, _stacklevel, dtype)


@register_model('pipline_test')
def pipline_test():
    GB['databuffer'].call_hooks('test', 'before_procedure')
    GB['databuffer'].call_hooks(
        'test', 'before_epoch')  # 就一轮，也写一下，方便训练验证时在wpoch时调用相同事件方便
    REGISTED_MODEL['ioperation']('pipline_test_epoch')
    GB['databuffer'].call_hooks('test', 'after_epoch')
    GB['databuffer'].call_hooks('test', 'after_procedure')


@register_model('pipline_test_epoch')
def pipline_test_epoch():  # test的epoch意味着不同数据集
    with torch.no_grad():
        datasink = GB['databuffer'].get_datasink('test')
        with alive_bar(len(datasink.dataloader),
                       title="pipline_test_epoch") as bar:
            for datasink.batch_inputs in datasink.dataloader:
                bar()
                GB['databuffer'].call_hooks('test', 'before_batch')
                if isinstance(datasink.batch_inputs['inputs'], dict):
                    GB['databuffer'].model.forward_prediction(
                        **{
                            k: v.to(GB['databuffer'].device)
                            for k, v in
                            datasink.batch_inputs['inputs'].items()
                        })
                else:
                    GB['databuffer'].model.forward_prediction(
                        tuple(
                            t.to(GB['databuffer'].device)
                            for t in datasink.batch_inputs['inputs']))
                GB['databuffer'].call_hooks('test', 'after_batch')


@register_model('pipline_inference')
def pipline_inference(texts):
    GB['databuffer'].inference_data = texts
    GB['databuffer'].call_hooks(
        'inference', 'before_epoch')  # 就一轮，也写一下，方便训练验证时在wpoch时调用相同事件方便
    REGISTED_MODEL['ioperation']('pipline_inference_epoch')
    GB['databuffer'].call_hooks('inference', 'after_epoch')
    datasink = GB['databuffer'].get_datasink('inference')
    res_data = []
    start = 0
    for t_len in datasink.truncation_step:
        for sample in datasink.get_sample()[start:start + t_len]:
            res_data.append(sample.pred_orig_truth)
            break
        start += t_len
    return res_data


@register_model('DatasinkBase')
class DatasinkBase(REGISTED_MODEL['DataBase']):
    """
    数据池，每个procedure都会有一个专属的数据池，里面存放 原始数据 sample ，dataset，dataloader
    每个batch的输入输出等
    也保存一些历史数据供实验报告和badcase导出使用

    注意：里面的数据一旦初始化后不能随意更改，通过self._freeze()来控制
    Args:
        REGISTED_MODEL (_type_): _description_
    """
    def __init__(self, sample_fn):
        self.sample_fn = sample_fn
        self.truncation_step = []
        self._samples = []
        self._dataset = None
        self._dataloader = None

        self.batch_inputs = None
        self.batch_loss = None  # 模型training forward的时候，loss需要传递给running_loss，用于strategy的优化
        self.batch_losses = None  # 模型training forward的时候，每个sample的单独loss，用于badcase分析
        self.batch_pred_ids = None

        # history_XXX存的是，这一个epoch中每个batch的历史数据，是一个list，用来badcase分析或者画图
        self.history_batch_loss = []
        self.history_batch_losses = []

        self.history_epoch_loss = []
        self.history_epoch_score = []

        self._freeze()

    def sweep_pred_attr(self):
        """
        清除每个sample的预测信息，进行下一轮训练验证
        """
        for sample in self._samples:
            sample.sweep_pred_attr()

    def get_sample(self, idx=None):
        """
        获取sample，有idx的时候根绝idx获取sample，没有idx则取所有的

        Args:
            idx (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if idx is None:
            return self._samples
        else:
            return self._samples[idx]

    def append_sample(self, **args):
        """
        往_sample里面添加sample
        """
        sample = REGISTED_MODEL['ioperation'](self.sample_fn, {})
        sample.set_values(**args)
        self._samples.append(sample)

    def create_sample(self, **args):
        """
        创建一个sample

        Returns:
            _type_: _description_
        """
        return REGISTED_MODEL['ioperation'](self.sample_fn, {**args})

    @property
    def dataset(self):
        return self._dataset

    def create_dataset(self, **args):
        """
        创建dataset
        """
        self._dataset = REGISTED_MODEL['ioperation']('DatasetBase', {**args})

    @property
    def dataloader(self):
        return self._dataloader

    def create_dataloader(self, **args):
        """
        创建dataloader
        """
        self._dataloader = DataLoader(**args)

    def replace_truncated_sample(self, truncated_samples):
        """
        为了替换新的sample，在需要做分割的场景，初始读入的sample是未经分割的
        所以在truncation环节要重新构建sample，然后替换原来的sample

        Args:
            truncated_samples (_type_): _description_
        """
        self._samples = []
        self.truncation_step = [len(x) for x in truncated_samples]
        for sample in truncated_samples:
            self._samples.extend(sample)


@register_model('HookProcessorBase')
class HookProcessorBase():
    """
    
    
    processor_fn: processor_bert_cls
    max_len: 512
    unk_token: "[UNK]"
    sep_token: "[SEP]"
    pad_token: "[PAD]"
    cls_token: "[CLS]"
    mask_token: "[MASK]"
    cls_token_segment_id: 1
    sequence_segment_id: 0
    pad_token_segment_id: 0
    vocab_path: '/app_name/nlp/azun/pretrained_models/bert-base/vocab.txt'
    
    weight_decay: 0.01
    learning_rate: 3e-5
    crf_learning_rate: 1e-3
    warmup_proportion: 0.1
    adam_epsilon: 1e-8
    label_fn: create_label_bios
    label_type: ${pipline.HookData.label_type}
    saving_vocab_path: 'model/vocab.txt'
    saving_vocab_fn: saving_vocab_base

    
    """
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        # GB['databuffer'].processor=REGISTED_MODEL['ioperation'](self.config.processor_fn,{
        #             **self.config
        #         })
    def before_epoch(self) -> None:
        if GB['databuffer'].procedure in ['training']:
            if GB['databuffer'].score_raising:

                REGISTED_MODEL['ioperation'](self.config.saving_vocab_fn, {
                    **self.config
                })

    def before_procedure(self) -> None:
        # 策略管理
        if GB['databuffer'].procedure in ['training']:
            GB['databuffer'].processor = REGISTED_MODEL['ioperation'](
                self.config.processor_fn, {
                    **self.config
                })
        elif GB['databuffer'].procedure in ['test', 'inference']:
            GB['databuffer'].processor = REGISTED_MODEL['ioperation'](
                self.config.processor_fn, {
                    **self.config, "saving_vocab_path":
                    self.config.saving_vocab_path
                })

    def after_procedure(self) -> None:
        # 训练结束，需要把model strategy_manager等清理掉，后面的test阶段需要自己重新载入
        if GB['databuffer'].procedure in ['training']:
            GB['databuffer'].processor = None

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('create_feature_sep_bert')
def create_feature_sep_bert():
    def auto_margin(max_len, seq_tokens, special_tokens_count):
        # 自适应长度，使切除最少的情况下满足512，主要应对 标题：文本这种长度相差大的情况
        # 平均一下不适合标题：文本这种长度相差大的情况
        # 从后往前切不适合 相似度，两个文本长度要一样的情况

        _sep_tokens = [[] for x in seq_tokens]
        _max_len = 0

        cursor = 0
        while _max_len <= max_len - special_tokens_count:
            if all([cursor >= len(tokens) for tokens in seq_tokens]):
                break
            for idx, tokens in enumerate(seq_tokens):
                if _max_len == max_len - special_tokens_count:
                    break
                if cursor >= len(tokens):
                    continue
                else:
                    _sep_tokens[idx].append(tokens[cursor])
                    _max_len += 1
            cursor += 1
        return _sep_tokens

    """
    bert输入的feature构建，seqs当中用SEP隔开,每个seq的长度是max_len除以seq的个数
    [CLS]seq[SEP]seq[SEP]seq[SEP]
    """
    mask_padding_with_zero = True
    a = True
    b = True

    from transformers import BertTokenizer

    # BertTokenizer.sep_token
    # BertTokenizer.se
    max_seq_length = GB['databuffer'].processor.max_len
    sep_token = GB['databuffer'].processor.sep_token
    sequence_segment_id = GB['databuffer'].processor.sequence_segment_id
    cls_token = GB['databuffer'].processor.cls_token
    cls_token_segment_id = GB['databuffer'].processor.cls_token_segment_id
    pad_token = GB['databuffer'].processor.pad_token
    pad_token_id = GB['databuffer'].processor.token2id(pad_token)
    pad_token_segment_id = GB['databuffer'].processor.pad_token_segment_id
    '''
    max_seq_length=GB['databuffer'].processor.max_len
    sep_token=GB['databuffer'].processor.sep_token
    sequence_segment_id=GB['databuffer'].processor.sequence_segment_id
    cls_token=GB['databuffer'].processor.cls_token
    cls_token_segment_id=GB['databuffer'].processor.cls_token_segment_id
    pad_token=GB['databuffer'].processor.pad_token
    pad_token_id=GB['databuffer'].processor.convert_tokens_to_ids(pad_token)
    pad_token_segment_id=GB['databuffer'].processor.pad_token_segment_id
    # label_map=GB['databuffer'].processor.label2id
'''
    features = []

    datasink = GB['databuffer'].get_datasink()
    for (ex_index, sample) in enumerate(datasink.get_sample()):

        seq_tokens = [
            GB['databuffer'].processor.text2token(text)
            for text in sample.sample_text
        ]

        # CLS  XXX SEP   XXX SEP XXX SEP   XXX SEP
        special_tokens_count = 1 + len(seq_tokens)
        seq_tokens = auto_margin(max_seq_length, seq_tokens,
                                 special_tokens_count)
        seq_tokens = [[token.lower() for token in tokens]
                      for tokens in seq_tokens]
        for t_idx, _tokens in enumerate(seq_tokens):
            if t_idx == 0:
                seq_tokens[t_idx] = [cls_token] + seq_tokens[t_idx]
            seq_tokens[t_idx] = seq_tokens[t_idx] + [sep_token]
        tokens = []
        # tokens=[cls_token]
        for _tokens in seq_tokens:
            tokens += _tokens

        label_id = GB['databuffer'].processor.label2id(
            sample.sample_label) if sample.sample_label is not None else None
        # CLS   TXTE_a  SEP   TEXT_B   SEP
        # print(seq_tokens)
        # print(len(seq_tokens[0]))
        # print(len(sample.sample_label))
        # print(len(label_id),'=====')
        if label_id is not None and len(label_id) != 1:
            label_id = label_id[:(max_seq_length - special_tokens_count)]
            label_id = [
                GB['databuffer'].processor.label2id('O')
            ] + label_id + [GB['databuffer'].processor.label2id('O')]

            # tokens+=[sep_token]
        segment_ids = []

        # segment_id  是  010101这样分的
        for idx, _tokens in enumerate(seq_tokens):
            if idx % 2 == 0:
                segment_ids += [0] * len(_tokens)
            else:
                segment_ids += [1] * len(_tokens)
        # segment_ids = [0] * (len(tokens_a))     +  [1] * (len(tokens_b))

        input_ids = GB['databuffer'].processor.token2id(tokens)

        input_mask = [1] * len(input_ids)  #[1,1,1,1,1,1]
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token_id
                      ] * padding_length  # #[101,1,2,3,4,102]  +[0,0,0,0,0]
        input_mask += [0] * padding_length  #[1,1,1,1,1,1]  +[0,0,0,0,0]
        segment_ids += [pad_token_segment_id
                        ] * padding_length  # [1,0,0,0,0,0]  +[0,0,0,0,0]
        if label_id is not None and len(label_id) != 1:
            label_id += [pad_token_id] * padding_length
        # print(len(label_id))

        sample_features = {
            'tokens': tokens,
            'token_ids': input_ids,
            'masks': input_mask,
            'lens': input_len,
            'segment_ids': segment_ids,
            'label_ids': label_id
        }
        sample.set_features(**sample_features)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

    datasink.create_dataset(samples=datasink.get_sample())

    REGISTED_MODEL['ioperation']('show_examples_feature', {})


@register_model('HookStrategyBase')
class HookStrategyBase():
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def after_batch(self) -> None:
        pass
        if GB['databuffer'].procedure in ['training']:
            GB['databuffer'].strategy.step()

    def before_procedure(self) -> None:
        if GB['databuffer'].procedure in ['training']:
            GB['databuffer'].strategy = REGISTED_MODEL['ioperation'](
                self.config.strategy_fn, {
                    **self.config
                })

    def after_procedure(self) -> None:
        if GB['databuffer'].procedure in ['training']:
            GB['databuffer'].processor = None

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('NetLinear')
class NetLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model('restore_base')
def restore_base():
    """
    用来将 id转为 truth
    """
    datasink = GB['databuffer'].get_datasink(GB['databuffer'].procedure)

    for sample in datasink.get_sample():
        sample.pred_sample_truth = GB['databuffer'].processor.id2truth(
            sample.pred_sample_label_id, sample.sample_text[0])
        sample.pred_orig_truth = GB['databuffer'].processor.id2truth(
            sample.pred_orig_label_id, sample.orig_text[0])


@register_model('SampleBase')
class SampleBase(REGISTED_MODEL['SampleObj']):
    """
    单条数据的基础类，有严格的数据存取限制：一旦赋值将不能变更，以此来保证数据不会在某个地方被更改，数据若是在某个地方被不小心更改，后期将很难查找出问题

    Args:
        REGISTED_MODEL (_type_): _description_
    """
    def __init__(self) -> None:

        # orig前缀：  原始
        # sample前缀: 处理后（切割，去噪等）

        # 本条数据的唯一编号，为了溯源
        self.guid = None

        # 原始数据
        self.orig_text = None
        # 切割/处理后的数据
        self.sample_text = None

        # # 原始数据
        # self.orig_text_a=None
        # # 切割/处理后的数据
        # self.sample_text_a=None

        # # 原始数据
        # self.orig_text_b=None
        # # 切割/处理后的数据
        # self.sample_text_b=None

        # 原始的GT，一般是中英文等有具体含义，如 实体名：人名、地点，类别名：正向、负向
        self.orig_truth = None
        self.sample_truth = None

        # 预测的GT，一般是中英文等有具体含义，如 实体名：人名、地点，类别名：正向、负向
        self.pred_orig_truth = None
        self.pred_sample_truth = None

        # 原始的truth映射的label，是truth和truth映射的id的中间表示，   truth==>label==>label_id  例如NER，则为 BIOS，  分类，则可为空或跟truth一致，
        self.orig_label = None
        self.sample_label = None

        # 预测的truth映射的label
        self.pred_orig_label = None
        self.pred_sample_label = None

        #  原始label映射的id
        self.orig_label_id = None
        self.sample_label_id = None

        #  预测的label映射的id
        self.pred_orig_label_id = None
        self.pred_sample_label_id = None

        self._features = None

        # 可以为每一个sample计算f1等指标
        self.orig_counter = None
        self.sample_counter = None

        self.orig_score = None
        self.sample_score = None

        self.loss = None

        self._freeze()

    def sweep_pred_attr(self):
        members = self.get_members()
        for member in members:
            if member.startswith('pred') or member in [
                    'loss', 'orig_counter', 'sample_counter', 'orig_score',
                    'sample_score', 'precision', 'score_info'
            ]:
                self.__dict__[member] = None

    def reset_value(self, key, value):
        self.__dict__[key] = value

    def set_values(self, **values):

        members = self.get_members()

        for k, v in values.items():
            if k not in members:
                raise AttributeError(f'发现未知属性:{k}，不允许新增，请检查将要赋予sample的属性值')
            setattr(self, k, v)

    def get_values(self):
        members = self.get_members()
        return {k: getattr(self, k) for k in members}

    def set_features(self, **values):
        self._features = {**values}

    def get_features(self, k=None):
        assert self._features is not None, 'features为空'
        if k is not None:
            return self._features[k]
        else:
            return self._features

    def get_members(self):
        members = [
            attr for attr in dir(self) if not callable(getattr(self, attr))
            and not attr.startswith("__") and not attr.startswith("_")
        ]
        return members

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        # for k,v in output.items():
        #     if isinstance(v,list) and all([isinstance(x,(int,str)) for x in v] ):
        #         # print(v )
        #         output[k]=' '.join([str(vv) for vv in v])
        #     elif isinstance(v,dict):
        #         for k1,v1 in v.items():
        #             if isinstance(v1,list) and all([isinstance(x,(int,str)) for x in v1] ):
        #                 output[k][k1]=' '.join([str(vv1) for vv1 in v1])
        return output


# json.dumps({'nums': arr}, cls=NpEncoder)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(),
                          indent=2,
                          sort_keys=True,
                          ensure_ascii=False,
                          cls=self.NpEncoder) + "\n"

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            return json.JSONEncoder.default(self, obj)


@register_model('saving_vocab_base')
def saving_vocab_base(**args):
    with open(args['saving_vocab_path'], 'w') as f:
        f.write('\n'.join(GB['databuffer'].processor.vocab))
    _ = f"保存词表: {args['saving_vocab_path']}"
    logger.info(_)
    # print(_), logger.info(_)


@register_model('saving_model_base')
def saving_model_base(**args):
    """
    保存模型
    """
    # os.makedirs(GB['databuffer'].saving_dir,exist_ok=True)
    save_path = os.path.join(args['save_dir'], 'model.pth')
    torch.save(GB['databuffer'].model.state_dict(), save_path)
    _ = f"保存完成:{save_path}"
    logger.info(_)
    # print(_), logger.info(_)


@register_model('HookModelBase')
class HookModelBase():
    def __init__(self, config) -> None:
        super().__init__()
        os.makedirs('model', exist_ok=True)
        self.config = config

    def before_batch(self) -> None:
        if not next(GB['databuffer'].model.parameters()
                    ).device == GB['databuffer'].device:
            GB['databuffer'].model.to(GB['databuffer'].device)
        if GB['databuffer'].procedure == 'training':
            if GB['databuffer'].model.training is not True:
                GB['databuffer'].model.train()
        else:
            if GB['databuffer'].model.training:
                GB['databuffer'].model.eval()
        if GB['databuffer'].procedure in ['training']:
            GB['databuffer'].model.zero_grad()

    def before_epoch(self) -> None:
        if GB['databuffer'].procedure == 'training':
            if GB['databuffer'].score_raising:
                REGISTED_MODEL['ioperation'](self.config.saving_model_fn, {
                    **self.config
                })
        GB['databuffer'].score_raising = False

    def before_procedure(self) -> None:
        if GB['databuffer'].procedure == 'training':
            # reload_param 是true的话需要重载参数
            GB['databuffer'].model = REGISTED_MODEL['ioperation'](
                self.config.model_fn, {
                    **self.config
                })
            REGISTED_MODEL['ioperation'](self.config.saving_model_fn, {
                **self.config
            })
            if 'reload_param' in self.config and self.config.reload_param:
                REGISTED_MODEL['ioperation'](
                    self.config.loading_bert_param_fn, {
                        **self.config, 'loading_path':
                        self.config.pretrained_bert_path
                    })
        elif GB['databuffer'].procedure in ['test', 'inference']:
            GB['databuffer'].model = REGISTED_MODEL['ioperation'](
                self.config.model_fn, {
                    **self.config
                })
            REGISTED_MODEL['ioperation'](self.config.loading_full_param_fn, {
                **self.config, 'loading_path':
                self.config.saving_model_path
            })

    def after_procedure(self) -> None:
        # 训练结束，需要把model strategy_manager等清理掉，后面的test阶段需要自己重新载入
        if GB['databuffer'].procedure == 'training':
            GB['databuffer'].model = None

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('DatabufferBase')
class DatabufferBase(REGISTED_MODEL['DataBase']):
    """
    生命周期存在于整个pipline，是所有数据流转，应用，更改的中间站，在lib的init中导入，全局可用，是中央控制器
    包括
    
    
    1. 训练的几大组件： 模型，优化策略，文本处理器 
    2. 训练/开发/测试 数据：_datasink_dic
    3. pipline运行时依赖的数据： loss，f1等
    4. 钩子
    5. 所有的hydra配置




    开发属性：
        唯一

    Args:
        REGISTED_MODEL (_type_): _description_
    """
    def __init__(self, config) -> None:
        self.config = config  # hydra的总config
        self.model = None  #  模型
        self.strategy = None  #优化策略
        self.processor = None  # 文本处理器
        self._datasink_dic = None  # 文本处理器
        self.epoch_count = -1  # epoch计数器
        self._hooks = []  #钩子
        self.inference_data = None  # 推理的数据，用于线上预测传值     databuffer.inference_data
        self.reaching_early_stop = False
        self.patience_count = 0
        self.datasink_fn = 'DatasinkBase' if not (
            'initialization' in config
        ) else config.templete.initialization.datasink_fn
        self.sample_fn = 'SampleBase' if not (
            'initialization' in config
        ) else config.templete.initialization.sample_fn
        self.score_raising = True  #指标是否上升，用来指导模型存储
        self._procedure = None
        self.init_datasink()
        # self.break_step=config.templete.common.break_step
        # self.epoch=config.templete.common.epoch
        # self.device=config.templete.common.device
        # 注册钩子

        for var_name, var_value in config.templete.hooks.items():
            setattr(self, var_name, var_value)

        for var_name, var_value in config.templete.common.items():
            setattr(self, var_name, var_value)

        if self.config.templete.common.break_step is not None and self.config.templete.common.epoch is None:
            self.epoch = self.break_step
        self._freeze()
        # 禁止添加新的属性，后面在代码中动态添加的这里看不到，会很难维护
    def init_datasink(self):
        self._datasink_dic = {
            "training":
            REGISTED_MODEL['ioperation'](self.datasink_fn, {
                'sample_fn': self.sample_fn
            }),
            "validation":
            REGISTED_MODEL['ioperation'](self.datasink_fn, {
                'sample_fn': self.sample_fn
            }),
            "test":
            REGISTED_MODEL['ioperation'](self.datasink_fn, {
                'sample_fn': self.sample_fn
            }),
            "inference":
            REGISTED_MODEL['ioperation'](self.datasink_fn, {
                'sample_fn': self.sample_fn
            })
        }

    def init_inference_datasink(self):
        self._unfreeze()
        self._datasink_dic['inference'] = REGISTED_MODEL['ioperation'](
            self.datasink_fn, {
                'sample_fn': self.sample_fn
            })
        self._freeze()

    @property
    def procedure(self):
        return self._procedure

    @procedure.setter
    def procedure(self, value):
        assert value in ['training', 'validation', 'test', 'inference']
        self._procedure = value

    def get_datasink(self, procedure=None):
        if procedure:
            return self._datasink_dic[procedure]
        else:
            return self._datasink_dic[self.procedure]

    def call_hooks(self, procedure, stage):
        for hook in self._hooks:
            self.procedure = procedure
            if hasattr(hook, stage):
                _ = f"procedure: {procedure}====stage: {stage}   {hook} "
                # print(_)#,logger.info(_)
                getattr(hook, stage)()

    def register_hooks_from_config(self, hooks):
        for hook_fn, hook_config in hooks.items():
            self._hooks.append(REGISTED_MODEL['ioperation'](
                hook_fn, {
                    'config': hook_config
                }))
            if hook_config is not None:
                for hook_config_name, hook_config_value in hook_config.items():
                    # print(hook_config_name)
                    if hook_config_name.endswith('_fn'):
                        METHOD_NAME_LIST.append(hook_config_value)


@register_model('HookLossBase')
class HookLossBase():
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def after_batch(self) -> None:
        if GB['databuffer'].procedure in ['training']:
            datasink = GB['databuffer'].get_datasink()
            datasink.batch_loss.backward()
            # GB['databuffer'].strategy.step()
            for idx, loss in zip(datasink.batch_inputs['idx'],
                                 datasink.batch_losses.cpu().numpy()):
                datasink.get_sample(idx).loss = loss
            datasink.history_batch_loss.append(
                datasink.batch_loss.detach().cpu().numpy())
            # print('batch_loss',datasink.batch_loss.detach().cpu().numpy())
            torch.nn.utils.clip_grad_norm_(GB['databuffer'].model.parameters(),
                                           1)

    def after_epoch(self) -> None:
        pass
        # if GB['databuffer'].procedure in ['training']:
        #     datasink=GB['databuffer'].get_datasink()
        #     datasink.history_epoch_loss.append(sum(datasink.history_batch_loss)/len(datasink.history_batch_loss))

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('HookTrainingBase')
class HookTrainingBase():
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.path = 'ootb'
        os.makedirs(self.path, exist_ok=True)

    def after_procedure(self) -> None:
        if "hook" in os.path.basename(__file__):
            if GB['databuffer'].procedure == 'inference':
                target = '.hydra/config.yaml'
                path = 'ootb/config.yaml'  # 已经创建好的
                if not os.path.exists(path):
                    shutil.copyfile(target, path)
                code = []
                # 直接调用，不用ioperation
                requirements = REGISTED_MODEL['ioperation']('gen_requirements',
                                                            {})
                requirements = [
                    x for x in requirements
                    if not re.findall('\s?from\slib\s', x)
                ]
                code += requirements
                code += REGISTED_MODEL['ioperation']('gen_funcation_code', {})
                file_head = []
                file_body = []
                file_tail = []
                area = 'head'
                with open(
                        os.path.join(get_original_cwd(),
                                     'deployment/training/training.py'),
                        'r') as f:
                    for line in f.readlines():
                        # code.append(line.strip('\n'))
                        if area == 'head':
                            file_head.append(line.strip('\n'))
                        elif area == 'head':
                            file_body.append(line.strip('\n'))
                        else:
                            file_tail.append(line.strip('\n'))
                        if area == 'head' and line.startswith('#'):
                            area = 'body'
                        if area == 'body' and line.startswith('if __name__'):
                            area = 'tail'
                code = file_head + code
                code = code + file_body
                code = code + file_tail
                with open(os.path.join(self.path, 'training.py'), 'w') as f:
                    f.write('\n'.join(code))
                print("gen_api_code 加载完毕"), logger.info('gen_api_code 加载完毕')

                # 写入一个example
                # datasink=GB['databuffer'].get_datasink(GB['databuffer'].procedure)
                # if isinstance(data_info,str):
                #     jsonl_file= list(jsonlines.open(os.path.join(data_info)))

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('create_dataloader')
def create_dataloader(batch_size=None,
                      shuffle=False,
                      sampler=None,
                      num_workers=0,
                      batch_sampler=None,
                      collate_fn=None,
                      pin_memory=False,
                      drop_last=False,
                      timeout=0,
                      worker_init_fn=None,
                      multiprocessing_context=None,
                      generator=None,
                      prefetch_factor=2,
                      persistent_workers=False):

    datasink = GB['databuffer'].get_datasink(GB['databuffer'].procedure)
    datasink.create_dataloader(
        dataset=datasink.dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # 测试的时候必须是负
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=REGISTED_MODEL[collate_fn],
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers)


@register_model('pipline_inference_epoch')
def pipline_inference_epoch():  # test的epoch意味着不同数据集
    with torch.no_grad():

        datasink = GB['databuffer'].get_datasink('inference')
        for datasink.batch_inputs in datasink.dataloader:
            GB['databuffer'].call_hooks('inference', 'before_batch')
            if isinstance(datasink.batch_inputs['inputs'], dict):
                GB['databuffer'].model.forward_prediction(
                    **{
                        k: v.to(GB['databuffer'].device)
                        for k, v in datasink.batch_inputs['inputs'].items()
                    })
            else:
                GB['databuffer'].model.forward_prediction(
                    tuple(
                        t.to(GB['databuffer'].device)
                        for t in datasink.batch_inputs['inputs']))
            GB['databuffer'].call_hooks('inference', 'after_batch')


@register_model('HookScoreF1')
class HookScoreF1():
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.past_max_score = -1

    # def before_epoch(self) -> None:

    def after_epoch(self) -> None:
        """
        两件事：
        1. 判断是否early stop
        2. 判断
        """
        if GB['databuffer'].procedure in ['validation', 'test']:
            datasink = GB['databuffer'].get_datasink()
            GB['databuffer'].score_raising = False
            REGISTED_MODEL['ioperation'](self.config.counter_fn)
            REGISTED_MODEL['ioperation'](self.config.metric_fn)

            if GB['databuffer'].procedure in ['validation']:
                # if len(datasink.history_epoch_score)==1:
                #     max_score=-1
                # else:
                #     max_score=max([x['f1'] for x in datasink.history_epoch_score])

                # current_score=datasink.history_epoch_score[-1]['f1']
                current_score = datasink.history_epoch_score[-1]['f1']
                if current_score > self.past_max_score:
                    _ = f"f1上升    :{self.past_max_score}====》:{current_score}"
                    logger.info(_)
                    # print(_), logger.info(_)
                    GB['databuffer'].patience_count = 0
                    GB['databuffer'].score_raising = True
                else:
                    GB['databuffer'].patience_count += 1
                    if GB['databuffer'].patience_count > GB['databuffer'].patience and GB['databuffer'].epoch_count > \
                            GB['databuffer'].min_patience_epoch:
                        GB['databuffer'].reaching_early_stop = True
                        _ = f"reaching early stop=====max_f1:{self.past_max_score}=====current_f1:{current_score}"
                        print(_), logger.info(_)

                self.past_max_score = current_score if current_score > self.past_max_score else self.past_max_score

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('NetDropout')
class NetDropout:
    def from_config(config):
        return torch.nn.Dropout(**config)


@register_model('pipline_training_epoch')
def pipline_training_epoch():  # 相同的数据，多跑几轮
    datasink = GB['databuffer'].get_datasink('training')
    with alive_bar(
            len(datasink.dataloader),
            title=f"pipline_training_epoch:{GB['databuffer'].epoch_count}"
    ) as bar:
        for batch_inputs in datasink.dataloader:
            datasink.batch_inputs = batch_inputs

            GB['databuffer'].call_hooks('training', 'before_batch')
            if isinstance(datasink.batch_inputs['inputs'], dict):
                GB['databuffer'].model.forward_training(
                    **{
                        k: v.to(GB['databuffer'].device)
                        for k, v in datasink.batch_inputs['inputs'].items()
                    })
            else:
                GB['databuffer'].model.forward_training(
                    tuple(
                        t.to(GB['databuffer'].device)
                        for t in datasink.batch_inputs['inputs']))
            GB['databuffer'].call_hooks('training', 'after_batch')
            bar()
            bar.text(f"loss:{datasink.batch_loss.item()}")


@register_model('save_running_example_base')
def save_running_example_base():
    datasink = GB['databuffer'].get_datasink('training')
    examples = [{'text': x.orig_text} for x in datasink.get_sample()[:3]]
    os.makedirs('ootb', exist_ok=True)
    with jsonlines.open('ootb/example.jsonl', mode='w') as writer:
        for x in datasink.get_sample()[:3]:
            writer.write({
                'text': x.orig_text,
            })
    _ = f"************* example ******************:"
    logger.info(_)
    # print(_), logger.info(_)
    print(examples, format=True), logger.info(_)


@register_model('HookServiceFlask')
class HookServiceFlask():
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

    def after_procedure(self) -> None:
        if "hook" in os.path.basename(__file__):
            if GB['databuffer'].procedure == 'inference':
                # self.parser_fn()
                self.copy_apibase(
                    os.path.join(get_original_cwd(), 'deployment/apibase'),
                    os.getcwd())
                code = []
                # 直接调用，不用ioperation
                code += REGISTED_MODEL['ioperation']('gen_requirements', {})
                code += REGISTED_MODEL['ioperation']('gen_funcation_code', {})

                code.extend([
                    "cfg = OmegaConf.load(os.path.join(os.getcwd(),'.hydra/config.yaml'))",
                    "GB['databuffer']=REGISTED_MODEL['ioperation']('DatabufferBase',{'config':cfg})",
                    "GB['databuffer'].register_hooks_from_config(cfg.templete.customer)",
                    "GB['databuffer'].call_hooks('inference','before_procedure')"
                ])

                with open('app/prediction.py', 'w') as f:
                    f.write('\n'.join(code))
                logger.info('gen_api_code 加载完毕')

    def copy_apibase(self, api_source_dir, target_dir):

        for src in os.listdir(api_source_dir):
            src_path = os.path.join(api_source_dir, src)
            print(f"HookServiceFlask-src_path   {src_path}")
            if os.path.isdir(src_path):
                shutil.copytree(src_path, os.path.join(target_dir, src))
            elif os.path.isfile(src_path):
                shutil.copy(src_path, target_dir)

    # def parser_fn(self):
    #     for hook_name,hook_info in GB['databuffer'].config.template.items():
    #         METHOD_NAME_LIST.append(hook_name)
    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('ProcessorClsBert')
class ProcessorClsBert():
    def __init__(self, **args) -> None:
        self.word2idx_dic = {}
        self.vocab = []
        with open(args['vocab_path']) as f:
            for idx, word in enumerate(f.readlines()):
                word = word.strip()
                self.word2idx_dic[word] = idx
                self.vocab.append(word)
        self.idx2word_dict = {v: k for k, v in self.word2idx_dic.items()}
        self.unk_token = args['unk_token']
        self.sep_token = args['sep_token']
        self.pad_token = args['pad_token']
        self.cls_token = args['cls_token']
        self.mask_token = args['mask_token']
        self.truth_types = args['truth_types'].split('|')
        self.label = self.initialize_labels(self.truth_types)
        self._id2label = {k: v for k, v in enumerate(self.label)}
        self._label2id = {v: k for k, v in enumerate(self.label)}
        self.max_len = args['max_len']
        self.sequence_segment_id = args['sequence_segment_id']
        self.cls_token_segment_id = args['cls_token_segment_id']
        self.pad_token_segment_id = args['pad_token_segment_id']
        self.do_lower_case = True
        _ = f"id2label: {self._id2label}",
        logger.info(_)
        # print(_), logger.info(_)
        _ = f"label2id: {self._label2id}",
        logger.info(_)
        # print(_), logger.info(_)

    def initialize_labels(self, truth_types):
        return truth_types

    def text2token(self, text):
        return list(text)

    def token2id(self, token):
        if isinstance(token, list):
            return [
                self.word2idx_dic.get(_token,
                                      self.word2idx_dic[self.unk_token])
                for _token in token
            ]
        else:
            return self.word2idx_dic.get(token,
                                         self.word2idx_dic[self.unk_token])

    def id2token(self, id):
        if isinstance(id, list):
            return [self.idx2word_dict.get(_id, self.unk_token) for _id in id]
        else:
            return self.idx2word_dict.get(id, self.unk_token)

    def id2label(self, id):
        if isinstance(id, list):
            return [self._id2label[_id] for _id in id]
        else:
            return self._id2label[id]

    def label2id(self, id):
        if isinstance(id, list):
            return [self._label2id[_id] for _id in id]
        else:
            return self._label2id[id]

    def truth2label(self, text, label):
        return label

    def label2truth(self, label):
        return label

    # TODO 删除?
    def tokenize(self, text):
        return list(text)

    def id2truth(self, id, text):
        return self.label2truth(self.id2label(id))

    def truth2id(self):
        pass


@register_model('pipline_train')
def pipline_train():
    GB['databuffer'].call_hooks('training', 'before_procedure')
    GB['databuffer'].call_hooks('validation', 'before_procedure')
    for GB['databuffer'].epoch_count in range(GB['databuffer'].epoch):

        GB['databuffer'].call_hooks('training', 'before_epoch')
        REGISTED_MODEL['ioperation']('pipline_training_epoch')
        GB['databuffer'].call_hooks('training', 'after_epoch')

        GB['databuffer'].call_hooks('validation', 'before_epoch')
        REGISTED_MODEL['ioperation']('pipline_validation_epoch')
        GB['databuffer'].call_hooks('validation', 'after_epoch')

        if GB['databuffer'].reaching_early_stop:
            break
    GB['databuffer'].call_hooks('validation', 'after_procedure')
    GB['databuffer'].call_hooks('training', 'after_procedure')


@register_model('count_tnpn_cls')
def count_tnpn_cls():
    datasink = GB['databuffer'].get_datasink(GB['databuffer'].procedure)
    for sample in datasink.get_sample():
        truth = sample.sample_truth
        founds = sample.pred_sample_truth
        rights = [x for x in founds if x in truth]
        truth_counter = Counter([x for x in truth])
        found_counter = Counter([x for x in founds])
        right_counter = Counter([x for x in rights])

        sample.sample_counter = {
            'truth_counter': truth_counter,
            'found_counter': found_counter,
            'right_counter': right_counter
        }

        truth = sample.orig_truth
        founds = sample.pred_orig_truth
        rights = [x for x in founds if x in truth]
        truth_counter = Counter([x for x in truth])
        found_counter = Counter([x for x in founds])
        right_counter = Counter([x for x in rights])
        sample.orig_counter = {
            'truth_counter': truth_counter,
            'found_counter': found_counter,
            'right_counter': right_counter
        }


@register_model('strategy_bert_base')
class strategy_bert_base:
    def __init__(self, **args) -> None:
        # param_optimizer = list(GB['databuffer'].model.named_parameters())

        bert_param_optimizer = list(
            GB['databuffer'].model.module_dict['bert'].named_parameters())
        linear_param_optimizer = list(
            GB['databuffer'].model.module_dict['linear'].named_parameters())

        total_step = len(GB['databuffer'].get_datasink(
            'training').dataloader) * GB['databuffer'].epoch
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in bert_param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01,
            'lr':
            args['lr']
        }, {
            'params': [
                p for n, p in bert_param_optimizer
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0,
            'lr':
            args['lr']
        }, {
            'params': [
                p for n, p in linear_param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01,
            'lr':
            args['linear_learning_rate']
        }, {
            'params': [
                p for n, p in linear_param_optimizer
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0,
            'lr':
            args['linear_learning_rate']
        }]
        warmup_steps = int(total_step * args['warmup_proportion'])
        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=args['lr'],
                               eps=args['adam_epsilon'])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_step)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()


@register_model('copy_dir')
def copy_dir(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for src in os.listdir(source_dir):
        src_path = os.path.join(source_dir, src)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, os.path.join(target_dir, src))
        elif os.path.isfile(src_path):
            shutil.copy(src_path, target_dir)
    _ = f"form {source_dir}  ====>   {target_dir}"
    logger.info(_)
    # print(_), logger.info(_)


@register_model('loss_ce')
def loss_ce(predict, label):
    """
    交叉熵loss

    Args:
        predict (_type_): _description_
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    return F.cross_entropy(predict, label), F.cross_entropy(predict.detach(),
                                                            label,
                                                            reduction='none')


@register_model('HookDataBase')
class HookDataBase():
    '''
    HookData:
    training_file_path: ${hyper.global.data_path}/loc/training.jsonl_
    validation_file_path: ${hyper.global.data_path}/loc/validation.jsonl_
    test_file_path: ${hyper.global.data_path}/loc/test.jsonl_
    inference_data: 

    create_sample_fn: sample_seq_pair
    create_label_fn: create_label
    create_feature_fn: create_feature_sep_bert
    create_dataloader: create_dataloader
    collate_fn: collate_bert
    restore_fn: restore_base
    '''
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def after_batch(self) -> None:
        if GB['databuffer'].procedure in ['test', 'validation', 'inference']:
            datasink = GB['databuffer'].get_datasink()
            for idx, pred_ids in zip(datasink.batch_inputs['idx'],
                                     datasink.batch_pred_ids):
                datasink.get_sample(idx).pred_sample_label_id = pred_ids
                datasink.get_sample(idx).pred_orig_label_id = pred_ids

    def before_epoch(self) -> None:
        if GB['databuffer'].procedure == 'inference':
            GB['databuffer'].init_inference_datasink()
            self.initializing_data(GB['databuffer'].inference_data, False)
        datasink = GB['databuffer'].get_datasink()
        datasink.sweep_pred_attr()

    def after_epoch(self) -> None:
        if GB['databuffer'].procedure in ['validation', 'test', 'inference']:
            REGISTED_MODEL['ioperation'](self.config.restore_fn)
        # if GB['databuffer'].procedure in ['inference']:
        #     GB['databuffer'].init_inference_datasink()

    def before_procedure(self) -> None:
        if GB['databuffer'].procedure == 'training':
            self.initializing_data(self.config.training_file_path,
                                   shuffle=True)

        elif GB['databuffer'].procedure == 'validation':
            self.initializing_data(self.config.validation_file_path,
                                   shuffle=False)

        elif GB['databuffer'].procedure == 'test':
            self.initializing_data(self.config.test_file_path, shuffle=False)

    def initializing_data(self, data_info, shuffle):
        REGISTED_MODEL['ioperation'](self.config.create_sample_fn, {
            "data_info": data_info
        })
        REGISTED_MODEL['ioperation'](self.config.create_label_fn, {
            **self.config
        })
        REGISTED_MODEL['ioperation'](self.config.create_feature_fn)
        REGISTED_MODEL['ioperation'](self.config.create_dataloader, {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "collate_fn": self.config.collate_fn
        })

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('HookBadcaseBase')
class HookBadcaseBase():
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.badcase_dir = 'report/badcase'
        os.makedirs(self.badcase_dir, exist_ok=True)

    def after_epoch(self) -> None:
        def flat(conditions):
            res = []
            for x in conditions:
                if isinstance(x, list):
                    res.extend(flat(x))
                else:
                    res.append(x)
            return res

        if GB['databuffer'].procedure in ['training', 'validation', 'test']:
            df_data = defaultdict(list)
            file_dir = os.path.join(self.badcase_dir,
                                    GB['databuffer'].procedure)
            os.makedirs(file_dir, exist_ok=True)
            datasink = GB['databuffer'].get_datasink(
                GB['databuffer'].procedure)
            for sample in datasink.get_sample():

                for k, v in sample.get_values().items():
                    if k in [
                            'guid', 'sample_text', 'sample_label',
                            'pred_sample_label', 'sample_label_id',
                            'pred_sample_label_id', 'sample_truth',
                            'pred_sample_truth'
                            'loss', 'f1', 'recall', 'precision'
                    ]:
                        # if v is not None :
                        if isinstance(v, list):
                            v = flat(v)
                            v = [str(x) for x in v]
                            v = '\n'.join(v)
                        df_data[k].append(v)

            df = pd.DataFrame(df_data)
            df.to_excel(
                os.path.join(
                    file_dir, 'badcase__' + GB['databuffer'].procedure + 'e_' +
                    str(GB['databuffer'].epoch_count) + '.xlsx'))

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('read_jsonline')
def read_jsonline(data_info):
    """
    适用于 seq2label,seqs2label,seq2labels,seqs2labels,

    首先取当前procedure对应的datasink，然后将读取的数据存入该datasink的samples中
    Args:
        data_info (str,list): data_info为字符串的时候，是jsonline的存储地址，否则则为传过来的数据
    """
    datasink = GB['databuffer'].get_datasink(GB['databuffer'].procedure)
    if isinstance(data_info, str):
        jsonl_file = list(jsonlines.open(os.path.join(data_info)))
        with alive_bar(len(jsonl_file), title=f"read jsonl_file") as bar:
            for idx, x in enumerate(jsonl_file):
                datasink.append_sample(
                    guid=x['guid'],
                    orig_text=[x['text']]
                    if isinstance(x['text'], str) else x['text'],
                    sample_text=[x['text']]
                    if isinstance(x['text'], str) else x['text'],
                    sample_truth=[x['truth']] if isinstance(
                        x['truth'], (str, int)) else x['truth'],
                    orig_truth=[x['truth']] if isinstance(
                        x['truth'], (str, int)) else x['truth'],
                )
                datasink.truncation_step.append(1)

                if GB['databuffer'].break_step == idx:
                    break
                bar()
    elif isinstance(data_info, (omegaconf.listconfig.ListConfig, list)):

        for idx, _text in enumerate(data_info):

            datasink.append_sample(
                guid=f'guid: {idx}',
                orig_text=[_text['text']]
                if isinstance(_text['text'], str) else _text['text'],
                sample_text=[_text['text']]
                if isinstance(_text['text'], str) else _text['text'],
                # orig_text = [_text] if isinstance(_text,str) else _text,
                # sample_text = [_text] if isinstance(_text,str) else _text,
                sample_truth=None,
                orig_truth=None)
            datasink.truncation_step.append(1)


@register_model('ActivateLogSoftmax')
class ActivateLogSoftmax(LogSoftmax):
    """
    激活函数: LogSoftmax, torch.nn.LogSoftmax
    

    Args:
        LogSoftmax (_type_): _description_
    """
    def __init__(self, dim) -> None:
        super().__init__(dim)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_model('loading_param_bert')
def loading_param_bert(strict=False, **args):
    """
    载入bert，注意bert的参数格式，这里的方法是在参数配置中去掉前面的bert.
    Args:
        strict (bool, optional): _description_. Defaults to False.
    """
    bert_state_dict = torch.load(args["loading_path"], map_location=GB['databuffer'].device)
    bert_state_dict = {
        re.sub('^bert.', '', k): v
        for k, v in bert_state_dict.items()
    }
    GB['databuffer'].model.module_dict['bert'].load_state_dict(
        bert_state_dict, strict)
    _ = f'载入参数 from {args["loading_path"]}'
    logger.info(_)
    # print(_), logger.info(_)


@register_model('HookInferenceBase')
class HookInferenceBase():
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.path = 'ootb'
        os.makedirs(self.path, exist_ok=True)

    def after_procedure(self) -> None:
        if "hook" in os.path.basename(__file__):
            if GB['databuffer'].procedure == 'inference':
                target = '.hydra/config.yaml'
                path = 'ootb/config.yaml'  # 已经创建好的
                if not os.path.exists(path):
                    shutil.copyfile(target, path)
                code = []
                # 直接调用，不用ioperation
                requirements = REGISTED_MODEL['ioperation']('gen_requirements',
                                                            {})
                requirements = [
                    x for x in requirements
                    if not re.findall('\s?from\slib\s', x)
                ]
                code += requirements
                code += REGISTED_MODEL['ioperation']('gen_funcation_code', {})
                file_head = []
                file_body = []
                file_tail = []
                area = 'head'
                with open(
                        os.path.join(get_original_cwd(),
                                     'deployment/inference/inference.py'),
                        'r') as f:
                    for line in f.readlines():
                        # code.append(line.strip('\n'))
                        if area == 'head':
                            file_head.append(line.strip('\n'))
                        elif area == 'head':
                            file_body.append(line.strip('\n'))
                        else:
                            file_tail.append(line.strip('\n'))
                        if area == 'head' and line.startswith('#'):
                            area = 'body'
                        if area == 'body' and line.startswith('#tail'):
                            area = 'tail'
                code = file_head + code
                code = code + file_body
                code = code + file_tail
                with open(os.path.join(self.path, 'inference.py'), 'w') as f:
                    f.write('\n'.join(code))
                print("gen_api_code 加载完毕"), logger.info('gen_api_code 加载完毕')

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


@register_model('HookMarkdownBase')
class HookMarkdownBase():
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.content = []
        self.markdown = REGISTED_MODEL['ioperation']("MarkdownBase", {}, {})

    def after_epoch(self) -> None:
        if GB['databuffer'].procedure == 'training':
            datasink = GB['databuffer'].get_datasink()

            self.markdown.draw_chart_line(
                title=
                f'loss_step_trends_within_epoch:{GB["databuffer"].epoch_count}',
                x_label='step',
                y_label='loss',
                x_data=list(range(len(datasink.history_batch_loss))),
                y_data={'loss': datasink.history_batch_loss})

    def after_procedure(self) -> None:
        if GB['databuffer'].procedure == 'validation':
            datasink = GB['databuffer'].get_datasink()
            self.markdown.draw_chart_line(
                title=f'overall_trends_{GB["databuffer"].procedure}',
                x_label='f1',
                y_label='epoch',
                x_data=list(range(len(datasink.history_epoch_score))),
                y_data={
                    # "loss": datasink.history_epoch_loss,
                    "f1": [x['f1'] for x in datasink.history_epoch_score],
                    "recall":
                    [x['recall'] for x in datasink.history_epoch_score],
                    "precision":
                    [x['precision'] for x in datasink.history_epoch_score],
                })

        if GB['databuffer'].procedure in ['inference']:
            self.markdown.write()

    def __repr__(self):
        return f"{os.path.abspath(__file__)}"


def stdout_write(msg: str):
    sys.stdout.write(msg)
    sys.stdout.flush()


def stderr_write(msg: str):
    sys.stderr.write(msg)
    sys.stderr.flush()


def line_print(*args, sep=' ', end='\n', file=None, flush=True, format=False):
    if format:
        args = (pformat(arg) for arg in args)
    else:
        args = (str(arg) for arg in args)
    if file == sys.stderr:
        stderr_write(sep.join(args))
    elif file in [sys.stdout, None]:
        # 获取被调用函数在被调用时所处代码行数
        line = sys._getframe().f_back.f_lineno
        # 获取被调用函数所在模块文件名
        file_name = sys._getframe(1).f_code.co_filename
        if format:
            stdout_write(
                f'{time.strftime("%H:%M:%S")}  "{file_name}:{line}"   {sep.join(args)} {end}'
            )  # 36  93 96 94
        else:
            stdout_write(
                f'{time.strftime("%H:%M:%S")}  "{file_name}:{line}"   {sep.join(args)} {end}'
            )  # 36  93 96 94
    else:
        print_raw(*args, sep=sep, end=end, file=file)


def print_exception(etype, value, tb, limit=None, file=None, chain=True):

    if file is None:
        file = sys.stderr
    for line in traceback.TracebackException(type(value),
                                             value,
                                             tb,
                                             limit=limit).format(chain=chain):
        # print(line, file=file, end="")
        if file != sys.stderr:
            stderr_write(f'{line} \n')
        else:
            stdout_write(f'{line} \n')


try:
    __builtins__.print = line_print
except AttributeError:
    """
    <class 'AttributeError'>
    'dict' object has no attribute 'print'
    """
    # noinspection PyUnresolvedReferences
    __builtins__['print'] = line_print


def init_logger():
    """
    :return: 初始化logger对象
    """
    if not LOG_PATH_LIST:
        logger = LogClass().singleLogger()
    else:
        logger = LogClass().multipleLogger(LOG_PATH_LIST)

    return logger


class LogClass(object):
    def __init__(self):
        self.base_path = os.path.join(PROJECT_PATH, LOG_BASE_PATH)
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self.fmt = '%(asctime)s PID:%(process)s %(levelname)s - %(filename)s [line:%(lineno)d]: %(message)s'

    def singleLogger(self):
        """
        :return: 返回全局的一个logger对象
        """
        log_path = os.path.join(self.base_path, 'log.log')

        sh = logging.StreamHandler()
        # 使用时间分片不是进程安全的
        # th = TimedRotatingFileHandler(filename=log_path, when='D', backupCount=30,
        #                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        ch = ConcurrentRotatingFileHandler(log_path,
                                           maxBytes=LOG_MAX_BYTES,
                                           backupCount=10,
                                           encoding='utf-8')
        # handlers = [sh, ch]
        handlers = [ch]
        logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'),
                            format=self.fmt,
                            handlers=handlers)

        logger = logging.getLogger(__name__)
        return logger

    def multipleLogger(self, loglist: list):
        """
        :param loglist: 传入配置的列表，name作为日志目录，
        :return: 返回多looger对象字典
        """
        log_dict = {}
        for name in loglist:
            log_sub_path = os.path.join(self.base_path, name)
            if not os.path.exists(log_sub_path):
                os.mkdir(log_sub_path)
            log_file = os.path.join(log_sub_path, f'{name}_run.log')

            mylogger = logging.getLogger(name)
            mylogger.setLevel(logging.INFO)

            # sh = logging.StreamHandler()
            # th = TimedRotatingFileHandler(filename=log_file, when='D', backupCount=30,
            #                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
            ch = ConcurrentRotatingFileHandler(log_file,
                                               maxBytes=LOG_MAX_BYTES,
                                               backupCount=10,
                                               encoding='utf-8')
            ft = logging.Formatter(self.fmt)
            # sh.setFormatter(ft)
            ch.setFormatter(ft)
            # mylogger.addHandler(sh)
            mylogger.addHandler(ch)
            log_dict[name] = mylogger
        return log_dict


# 日志模块
logger = init_logger()


class MysqlClient(object):
    """
    作为mysql的客户端连接数据库
    """
    def __init__(self,
                 host,
                 port,
                 user='indeednlp',
                 passwd='XcNKKbudjociXKl5U',
                 database='indeednlp_service',
                 charset='utf8mb4',
                 mincached=10,
                 maxcached=20,
                 maxshared=10,
                 maxconnections=200,
                 blocking=True,
                 maxusage=100,
                 setsession=None,
                 reset=True):
        """
        :param host: 连接数据库主机地址
        :param port: 连接数据库主机端口，默认3306
        :param user: 连接数据库用户名
        :param passwd: 连接数据库密码
        :param database: 连接数据库库名
        :param charset: 字符集
        :param mincached: 连接池中空闲连接的初始数量
        :param maxcached: 连接池中空闲连接的最大数量
        :param maxshared: 共享连接的最大数量
        :param maxconnections: 创建连接池的最大数量
        :param blocking: 超过连接数时状态，True是等待， False是报错
        :param maxusage: 单个连接的最大重复使用次数
        :param setsession: 可选的数据库命令列表
        :param reset:  数据库回滚安全模式
        """
        try:
            self._Pool = PooledDB(pymysql,
                                  host=host,
                                  port=port,
                                  user=user,
                                  password=passwd,
                                  database=database,
                                  charset=charset,
                                  mincached=mincached,
                                  maxcached=maxcached,
                                  maxshared=maxshared,
                                  maxconnections=maxconnections,
                                  maxusage=maxusage,
                                  blocking=blocking,
                                  setsession=setsession,
                                  reset=reset)
        except pymysql.Error as e:
            traceback.print_exc(e)
            logger.info(f"create mysqlpool at {host}:{port} failed")

    def create_connect(self):
        """
        :return: 返回数据库连接对象和游标对象
        """
        # 从数据库连接池中获取一个连接
        conn = self._Pool.connection()
        # 创建游标
        # cursor = conn.cursor()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        return conn, cursor

    def close_coonect(self, conn, cursor):
        """
        :param coon: 连接对象
        :param cursor: 游标对象
        :return: 释放资源
        """
        cursor.close()
        conn.close()

    def excute_command(self, sql, *args):
        """
        :param sql: 要被执行的sql语句
        :param args: 可能被传递的参数
        :return: 返回结果，除了查询语句，其他语句返回结果均为 OK
        """
        conn, cursor = self.create_connect()

        try:
            cursor.execute(sql)
            logger.info(f"excute `{sql}` success")
            conn.commit()

            if "select" in sql:
                res = json.dumps(cursor.fetchall())
                logger.info(f" {res}")
                return res
            else:
                logger.info("OK")
                return "OK"

        except pymysql.Error as e:
            traceback.print_exc(e)
            logger.info(f"excute {sql} failed")
            conn.rollback()
            logger.info("mysql rollback success")
        finally:
            self.close_coonect(conn, cursor)


db_config = {
    "host": "10.1.0.247",
    "port": 3306,
    "user": "indeednlp",
    "password": "XcNKKbudjociXKl5U",
    "db": "indeednlp_service"
}
# mysqldb = MysqlClient(host= db_config["host"], port= db_config["port"], user= db_config["user"],  passwd=db_config["password"],
#                           database=db_config["db"])

#tail
cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), 'config.yaml'))
# cfg.templete.common.data_info = json.load(
#     open(os.path.join(os.getcwd(), 'data/orig/', 'data_info.json')))
cfg = OmegaConf.to_container(cfg, resolve=False)
cfg = OmegaConf.create(dict(cfg))
GB['databuffer'] = REGISTED_MODEL['ioperation']('DatabufferBase', {
    'config': cfg
})
GB['databuffer'].register_hooks_from_config(cfg.templete.hooks)
GB['databuffer'].call_hooks('inference', 'before_procedure')


def bert_predict(text):
    examples = [text]
    inference_data = REGISTED_MODEL['ioperation']('pipline_inference', {
        'texts': examples
    })
    # print(inference_data)
    pred_label = inference_data[0][0]
    # probs = inference_data[0][1].data.cpu().numpy().max()
    # prob = torch.nn.functional.softmax(probs)[int(pred_label[0])].data.cpu().numpy()
    return pred_label


if __name__ == '__main__':
    # examples=list(jsonlines.open('examples.jsonl'))
    # inference_data=REGISTED_MODEL['ioperation']('pipline_inference',{'texts':examples})
    res = bert_predict({
        "text": [
            "授权委托书任勇系山东莱芜烟草有限公司法定代表人,现授权委托烟叶生产经营科王传胜为本公司合法代理人,在2021年烟叶种植保险业务范围内,以山东莱芜烟草有限公司的名义与中国人民财产保险股份有限公司莱芜市分公司审核签订合同。2代理人应该严格按照委托权限行使代理权,不得转让代理权。该授权委托有效期自20216至2021年7月9日止特此委托。委托单位:山东莱莞烟草有限公司(盖章)法定代表被授权人:371日期:"
        ]
    })
    print(res)