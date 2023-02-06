#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :logClass.py
# @Time      :2021/9/28 11:27
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)

import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler
from concurrent_log_handler import ConcurrentRotatingFileHandler
from config.config import Config


def init_logger():
    """
    :return: 初始化logger对象
    """
    if not Config.LOG_PATH_LIST:
        logger = LogClass().singleLogger()
    else:
        logger = LogClass().multipleLogger(Config.LOG_PATH_LIST)

    return logger


class LogClass(object):

    def __init__(self):
        self.base_path = os.path.join(Config.PROJECT_PATH, Config.LOG_BASE_PATH)
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self.fmt = '%(asctime)s PID:%(process)s %(levelname)s - %(filename)s [line:%(lineno)d]: %(message)s'

    def singleLogger(self):
        """
        :return: 返回全局的一个logger对象
        """
        log_path = os.path.join(self.base_path, 'run.log')

        sh = logging.StreamHandler()
        # 使用时间分片不是进程安全的
        # th = TimedRotatingFileHandler(filename=log_path, when='D', backupCount=30,
        #                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器

        ch = ConcurrentRotatingFileHandler(log_path, maxBytes=Config.LOG_MAX_BYTES, backupCount=10, encoding='utf-8')
        handlers = [sh, ch]
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

            sh = logging.StreamHandler()
            # th = TimedRotatingFileHandler(filename=log_file, when='D', backupCount=30,
            #                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器

            ch = ConcurrentRotatingFileHandler(log_file, maxBytes=Config.LOG_MAX_BYTES, backupCount=10, encoding='utf-8')

            ft = logging.Formatter(self.fmt)
            sh.setFormatter(ft)
            ch.setFormatter(ft)

            mylogger.addHandler(sh)
            mylogger.addHandler(ch)
            log_dict[name] = mylogger

        return log_dict


# 日志模块
logger = init_logger()

if __name__ == '__main__':
    # 全局logger测试
    loggerobj = LogClass()
    logger = loggerobj.singleLogger()
    logger.info(f"logger:{id(logger)}")
    for i in range(30):
        logger.info(f"hello, i am singleLogger{i}")
        time.sleep(1)

    # 多logger对象测试
    # loggerobj = LogClass()
    # logger = loggerobj.multipleLogger(Config.LOG_PATH_LIST)
    # for i in range(5):
    #     for key, value in logger.items():
    #         value.info(f"hello, i am {value}{i}")
    #         time.sleep(1)
