#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2021/9/14 10:44
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)

__all__ = ['config']

import yaml
import argparse
import omegaconf
from omegaconf import OmegaConf
import os
from prettyprinter import cpprint
import shutil

arg_parser = argparse.ArgumentParser(add_help=False)
# arg_parser.add_argument('config_path', type=str, default='config/config_dev.yml',required=False)
args, _ = arg_parser.parse_known_args()
f = open('config/config_dev.yml', 'r', encoding='utf-8')
# cont = f.read()
CONFIG_PARAM = OmegaConf.load(f)
CONFIG_PARAM = omegaconf.DictConfig(CONFIG_PARAM)

parent_path = os.path.dirname(os.path.dirname(__file__))
#
CONFIG_PARAM['MODEL']['modelpath_NER'] = os.path.join(parent_path, CONFIG_PARAM['MODEL']['modelpath_NER'])
# CONFIG_PARAM['MODEL']['modelpath_NER'] = '"/app_name/nlp/azun/supertext_auto/data/exp/model/yanyi-zaozhuangbiaoshu/chinese_roberta_L-4_H-512_span_枣庄标书+_v4'
