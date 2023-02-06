#!/bin/sh

cd /opt/project/apibaseproject/

echo '当前目录:'
echo `pwd`

export LC_ALL=en_US.UTF-8
export PYTHONPATH=$PYTHONPATH:`pwd`

# 直接起服务
/usr/bin/python3  setup_app.py

# 用uwsgi部署
#/usr/local/python3/bin/uwsgi --ini ./docker/base/uwsgi.ini

# 用gunicorn 部署,每个显卡启动一个端口，然后用nginx做负载均衡
#/usr/local/python3/bin/gunicorn -c docker/gunicorn.conf -b 0.0.0.0:5001 setup_app:flask_app