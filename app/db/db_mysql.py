#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :db_mysql.py
# @Time      :2021/10/13 18:01
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)


import json
import pymysql
import traceback
from utils.logClass import logger
from dbutils.pooled_db import PooledDB


class MysqlClient(object):
    """
    作为mysql的客户端连接数据库
    """

    def __init__(self, host, port, user='root', passwd=None, database=None, charset='utf8',
                 mincached=10, maxcached=20, maxshared=10, maxconnections=200, blocking=True,
                 maxusage=100, setsession=None, reset=True):
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
            self._Pool = PooledDB(
                pymysql,
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
                reset=reset
            )
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
        cursor = conn.cursor()
        # cursor = coon.cursor(pymysql.cursors.DictCursor)
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


# 当需要全局使用mysql数据库时打开，这样保证只创建一个数据库连接池，所需要的数据库连接从连接池获取
# with open("./db_config.json") as f:
#     db_config = json.load(f)
# mysqldb = MysqlClient(db_config["mysql"]["host"], db_config["mysql"]["port"], passwd=db_config["mysql"]["password"],
#                       database=db_config["mysql"]["db"])

if __name__ == '__main__':
    with open("db_config.json") as f:
        db_config = json.load(f)
    mysqldb = MysqlClient(db_config["mysql"]["host"], db_config["mysql"]["port"], passwd=db_config["mysql"]["password"],
                          database=db_config["mysql"]["db"])

    # 创建表
    # mysqldb.excute_command(
    #     "create table if not exists user ("
    #     "`id` int primary key  AUTO_INCREMENT KEY COMMENT '用户编号',"
    #     "`username` varchar(30) not null COMMENT '用户名'"
    #     ")ENGINE=INNODB DEFAULT CHARSET=utf8;"
    # )

    # mysqldb.excute_command("CREATE TABLE IF NOT EXISTS user ("
    #                   "`id` INT primary key AUTO_INCREMENT KEY COMMENT '用户编号',"
    #                   "`username` VARCHAR(20) NOT NULL UNIQUE COMMENT '用户名',"
    #                   "`password` CHAR(32) NOT NULL COMMENT '密码',"
    #                   "`email` VARCHAR(50) NOT NULL UNIQUE COMMENT '邮箱',"
    #                   "`age` TINYINT UNSIGNED NOT NULL DEFAULT 18 COMMENT '年龄',"
    #                   "`sex` ENUM('man','woman','baomi') NOT NULL DEFAULT 'baomi' COMMENT '性别',"
    #                   "`tel` CHAR(11) NOT NULL UNIQUE COMMENT '电话',"
    #                   "`addr` VARCHAR(50) NOT NULL DEFAULT 'beijing' COMMENT '地址',"
    #                   "`card` CHAR(18) NOT NULL UNIQUE COMMENT '身份证号',"
    #                   "`married` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '0代表未结婚，1代表已结婚',"
    #                   ")ENGINE=INNODB DEFAULT CHARSET=UTF8;")

    # 查
    mysqldb.excute_command('select * from user;')

    # 增
    # mysqldb.excute_command('insert into user values (2, "jerry"), (3, "marry")')

    # 改
    # mysqldb.excute_command('update user set username="jerry" where id=2')

    # 删
    # mysqldb.excute_command('delete from user where id=2;')
