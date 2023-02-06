#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :db_mongodb.py
# @Time      :2021/10/14 16:47
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)

import json
from pymongo import MongoClient


class MongoObj(object):
    def __init__(self, host, port=27017, db=None, collection=None):
        self.mongo = MongoClient(host, port)
        self.db = self.mongo[db]
        self.collection = self.db[collection]

    def mongo_query(self):
        query = json.dumps(self.collection.find().limit(100), ensure_ascii=False).encode('utf-8')


if __name__ == '__main__':
    pass
