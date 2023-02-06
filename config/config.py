#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :config.py.py
# @Time      :2021/9/14 10:44
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)
import json
import os
import time


class Config(object):
    """用于app加载全局变量信息
    :return:
    """
    # 配置项目所在路径
    PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # 配置当前模式
    DEBUG = False

    # 配置日志的根目录，初始化logger对象时创建
    LOG_BASE_PATH = "log"

    # 配置多日志对象的，默认为空，列表元素对应log目录下子目录
    LOG_PATH_LIST = None
    # LOG_PATH_LIST = ['nlp', 'ocr', 'cv']

    # 配置log存储的时间格式，默认为按天分割
    LOG_TIME = time.strftime('%Y%m%d', time.localtime(time.time()))

    # 配置log存储的最大限制，按照日志大小分割，默认10M
    LOG_MAX_BYTES = 1024 * 1024 * 10

    # 指定页面展示可提供服务的接口
    request_url = ["http://127.0.0.1:5001/nlp/", "http://127.0.0.1:5001/nlp/interface"]

    # 根据不同接口可提供一个样例
    examples = json.dumps({
        "http://127.0.0.1:5001/nlp/": {"name": "i am tom"},
        "http://127.0.0.1:5001/nlp/interface": {"name": "i am jerry"}
    })

