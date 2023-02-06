#FROM registry.ii-ai.tech/nlp/python:base
#FROM registry.ii-ai.tech/nlp/nlp_image_clean:base
FROM registry-deqing.ii-ai.tech/nlp/laiwu_nlp:base

# 切换工作目录
WORKDIR /opt/project/apibaseproject/

# 拷贝项目代码
COPY . .

# 配置文件
ENV LC_ALL=en_US.UTF-8

# 再装一次依赖，避免更新依赖后需要重新打基础包
#RUN yum install -y dmidecode
RUN pip3 install -U pip -i https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN chmod 755 ./docker/run.sh

CMD sed -i -e 's/\r$//' ./docker/run.sh \
    && chmod 755 ./docker/run.sh \
    && /bin/bash -c ./docker/run.sh

