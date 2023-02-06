#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2021/9/26 10:23
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)


from flask_sqlalchemy import SQLAlchemy
import setup_app

db = SQLAlchemy(setup_app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120), unique=True)

    def __repr__(self):
        return '<User %s>' % self.username

