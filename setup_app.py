# -*- coding:utf-8 -*-
'''
Author       : azun
Date         : 2022-05-30 17:03:54
LastEditors: yanyi
LastEditTime: 2022-12-09 10:34:28
'''
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :setup_app.py.py
# @Time      :2021/9/14 10:38
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)


from app import create_app

flask_app = create_app()

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=16224, debug=False)
