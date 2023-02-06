#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :db_redis.py
# @Time      :2021/10/12 10:03
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)
import json

import redis
import time
from config.config import Config
from utils.logClass import logger


class RedisObj(object):
    """
    创建redis连接
    """
    def __init__(self, host, port, db=0):
        """
        :param host: 主机
        :param port: 端口
        :param db: 指定db
        """
        self.pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True)
        self.redis = redis.Redis(connection_pool=self.pool)
        # self.pipe = self.redis.pipeline(transaction=False)
        # 默认的情况下，管道里执行的命令可以保证执行的原子性，执行pipe = r.pipeline(transaction=False)可以禁用这一特性。
        self.pipe = self.redis.pipeline()  # 创建一个管道

    def redis_set(self, t, *args, name=None, ex=None, px=None, nx=False, xx=False):
        if t == "str":
            self.redis.mset(*args)
            logger.info(f"set str {args} success")
        if t == "hash":
            self.redis.hset(name, *args)
            logger.info(f"set hash {args} success")
        if t == "list" and args[0] == "l":
            self.redis.lpush(name, *args[1::])
            logger.info(f"set list lpush {args[1::]} success")
        if t == "list" and args[0] == "r":
            self.redis.rpush(name, *args[1::])
            logger.info(f"set list rpush {args[1::]} success")
        if t == "set":
            self.redis.sadd(name, *args)
            logger.info(f"set set add {args} success")
        if t == "zset":
            self.redis.sadd(name, *args)
            logger.info(f"set zset add {args} success")

    def redis_get(self, t, key, name=None, index=None):
        res = 0
        if t == "str":
            res = self.pipe.get(key).execute()
            logger.info(f"get str {key} success, result is {res}")
        if t == "hash":
            res = self.pipe.hset(name, key).execute()
            logger.info(f"get hash {key} success, result is {res}")
        if t == "list":
            res = self.pipe.lindex(name, index).execute()
            logger.info(f"get list index {index} success, result is {res}")
        if t == "set":
            res = self.pipe.smembers(name).execte()
            logger.info(f"get set members {name} success, result is {res}")
        return res


if __name__ == '__main__':
    with open("db_config.json") as f:
        db_config = json.load(f)
    r = RedisObj(db_config["redis"]["host"], db_config["redis"]["port"])
    r.redis_set("str", {"szzn": "ai", "name": "yize"})
    res = r.redis_get("str", "name")
    logger.info(f"获取res:{res} OK")
