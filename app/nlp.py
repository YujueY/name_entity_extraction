'''
Author: yanyi
Date: 2022-11-22 10:38:55
LastEditors: yanyi
LastEditTime: 2022-12-22 14:13:35
FilePath: /laiwu_tobacco_xunjia/app/nlp.py
'''
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :nlp.py
# @Time      :2021/9/16 9:27
# @Author    :伊泽瑞尔(liuyongjun@i-i.ai)

from flask import Blueprint, render_template, request, current_app
from utils.logClass import logger
from config.config import Config
from config import CONFIG_PARAM, parent_path
from flask import request, Flask
import json
from copy import deepcopy
from prettyprinter import cpprint
# from controller.model_inference import ModelInference
from controller.extraction import ModelInference
from prettyprinter import cpprint, pformat
import traceback
import os
import time

nlp_logger = logger

nlp_bp = Blueprint('nlp', __name__, url_prefix='/')

model = ModelInference(max_seq_length=512, checkpoint=CONFIG_PARAM['MODEL']['modelpath_NER'])
# model = ModelInference()


@nlp_bp.route("/", methods=['GET', 'POST'])
def nlp_index():
    if request.method == 'GET':
        nlpconfig = current_app.config
        nlp_logger.info(f"nlpconfig:{nlpconfig}")
        return render_template('index.html')

    elif request.method == 'POST':
        nlp_logger.info(f"request.method:{request.method}")
        nlp_logger.info(f"request.cookies:{request.cookies}")
        res = request.json
        nlp_logger.info(f"from client json is {res}")
        res_args = request.args
        nlp_logger.info(f"from client res_args is {res_args}")
        res_form = request.form
        nlp_logger.info(f"from client res_form is {res_form}")

        return 'OK'


def get_review_result(json_data):
    nlp_logger.info('*' * 40)
    nlp_logger.info('json_data')
    nlp_logger.info(json_data['texts'][:20])
    nlp_logger.info('*' * 40)

    text = json_data['texts']
    lines_info_data = json_data['lines_info_data']
    id = json_data['id']
    all_item = json_data['all_item']
    # ner_label_dic = deepcopy(ner_label_dic_templet)
    # model.labels_inference(text, ner_label_dic)
    # review(ner_label_dic, text)
    # review_results, score_result = json_reformat(doc_class,ner_label_dic)
    review_results = model.labels_inference(text, lines_info_data, all_item, id)
    print(review_results)
    return review_results


@nlp_bp.route('/laiwu_tobatoo_xunjia', methods=["GET", "POST"])
def tender_review():
    try:
        data = request.get_data()
        json_data = json.loads(data.decode('utf-8'))
        review_results = get_review_result(json_data)
        print(review_results)
        return json.dumps({"review_results": review_results, "status": 0},
                          ensure_ascii=False)
    except Exception as e:
        nlp_logger.info(str(traceback.print_exc()))

        return json.dumps({"review_results": {}, "score_result": {}, "status": 4}, ensure_ascii=False)




if __name__ == '__main__':
    pass