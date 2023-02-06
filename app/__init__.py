#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2021/9/14 10:37
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)

import datetime
from flask import Flask
from flask_cors import CORS
from app.nlp import nlp_bp
from config.config import Config
from utils.logClass import init_logger


# from flask_sqlalchemy import SQLAlchemy

def create_app():
    """
    :return: return a app object
    """
    app = Flask(__name__)
    CORS(app)

    # 初始化配置
    # app.config.from_file('config.py')
    app.config.from_object(Config)

    # 注册蓝图
    app.register_blueprint(nlp_bp)

    # 配置数据库连接
    # db = SQLAlchemy(app)

    # app的中间件部分，nginx + gunicorn处理高并发

    @app.route("/heartbeat")
    def heartbeat():
        """心跳检测api，用于运维对服务的心跳检测
        :return: 返回检测时间
        """
        return "{}".format(datetime.datetime.now())

    return app


if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(host='0.0.0.0', port=5001, debug=False)
