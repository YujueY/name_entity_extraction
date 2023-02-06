#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :source_watch.py
# @Time      :2021/10/22 10:27
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)
import time

import psutil
import matplotlib.pyplot as plt
from paramiko_ssh import ParamikoClient
from utils.logClass import logger


def get_info():
    cpu_per = psutil.cpu_percent()
    mem_per = psutil.virtual_memory().used / psutil.virtual_memory().total * 100
    io_count = psutil.disk_io_counters()
    return cpu_per, mem_per, io_count


client = ParamikoClient("101.69.161.66", port=6719, user="shizai", passwd="shizaiAi170N_249")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.ion()
plt.figure(figsize=(10, 10))
cpu = []
mem = []
io = []
times = []

while True:
    t = time.strftime("%H:%M:%S", time.localtime())
    result = client.excute_cmd(
        "docker ps | grep api_base | awk '{print $1}' | xargs -I {} docker exec {} bash -c 'free' "
        "| grep Mem | awk '{print $2,$3}'").split(" ")
    logger.info(f"当前获取到的容器内存总量：{result[0]}，使用量：{result[1]}")
    mem_per = float(result[1]) / float(result[0]) * 100
    times.append(t)
    # cpu.append(cpu_per)
    mem.append(mem_per)
    logger.info(f"当前获取到的容器内存占比：{mem_per}")
    # io.append(io_count)

    # cx = plt.subplot(221)
    # plt.plot(times, cpu, label='CPU', color="b")
    # plt.ylabel("CPU 使用率 %", fontsize=14)
    # plt.xticks(rotation=6, fontsize=8)
    # plt.yticks(range(0, 110, 10))

    mx = plt.subplot()
    plt.plot(times, mem, label='MEM', color="g")
    plt.ylabel("MEM 使用率 %", fontsize=14)
    plt.xticks(rotation=6, fontsize=8)
    plt.yticks(range(0, 100, 10))

    # ix = plt.subplot(212)
    # plt.plot(times, io, label='IO', color="y")
    # plt.ylabel("IO 使用率 %", fontsize=14)
    # plt.xticks(rotation=6, fontsize=8)
    # plt.yticks(range(0, 110, 10))

    plt.pause(1)
    plt.ioff()
    plt.show()
