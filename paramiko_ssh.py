#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :paramiko_ssh.py
# @Time      :2021/10/22 14:45
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)


import paramiko
from utils.logClass import logger
from paramiko.ssh_exception import NoValidConnectionsError, AuthenticationException


class ParamikoClient(object):
    """
    创建一个远程链接服务器的客户端
    """

    def __init__(self, host, port, user, passwd):
        """
        :param host: 远程连接的主机
        :param port: 端口
        :param user: 用户
        :param passwd: 密码
        """
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        try:
            logger.info(f"正在连接:{host}......")
            self.client.connect(hostname=host, port=port, username=user, password=passwd)
        except NoValidConnectionsError:
            logger.info("connection failed, please check the right host")
        except AuthenticationException:
            logger.info("authentication failed, please check the right password")

    def excute_cmd(self, cmd, is_close=False):
        """
        :param cmd: 要被执行的命令
        :param is_close: 标识符，设置为Ture客户端关闭
        :return: 返回的结果
        """
        try:
            stdin, stdout, stderr = self.client.exec_command(cmd)
            result = stdout.read().decode("utf-8") if stdout else None
            # logger.info(f"excute {cmd} success, result is {result}")
            return result
        except Exception as e:
            logger.info("paramiko excute command failed")
            is_close = True
        finally:
            if is_close is True:
                self.client.close()


if __name__ == '__main__':
    client = ParamikoClient("101.69.161.66", port=6719, user="shizai", passwd="shizaiAi170N_249")
    result = client.excute_cmd(
        "docker ps | grep api_base | awk '{print $1}' | xargs -I {} docker exec {} bash -c 'ps -ef' | grep python3")
    logger.info(result)

    result2 = client.excute_cmd(
        "docker ps | grep api_base | awk '{print $1}' | xargs -I {} docker exec {} bash -c 'free' "
        "| grep Mem | awk '{print $3}'", is_close=True)
    logger.info(result2)

    # result3 = client.excute_cmd("docker ps | grep api_base | awk '{print $1}'")
    # logger.info(result3)

