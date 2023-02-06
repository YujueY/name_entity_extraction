# -*- coding: utf-8 -*-
'''
@version V1.0.0
@Time : 2021/11/12 9:48 上午 
@Author : azun
@File : extraction.py 
'''
import regex as re
from inference.span_1 import GeneralModelSpanPredict
from utils.time_cost import time_cost
from controller.inference import bert_predict
from utils.logClass import logger
from prettyprinter import pformat
from typing import List
import traceback
from controller.chinese_numerals2Arabic_numerals import cn2ara
from controller.time_Norm.TimeNormalizer import TimeNormalizer
from config import CONFIG_PARAM
import requests
import json
import threading
m_lock = threading.Lock()

cnn = [
    '〇', '一', '二', '三', '四', '五', '六', '七', '八', '九', '零', '壹', '贰', '叁', '肆',
    '伍', '陆', '柒', '捌', '玖', '貮', '两', '十', '拾', '百', '佰', '千', '仟', '万', '萬',
    '亿', '億', '兆'
]
import regex as re

pattern_str = '((?P<nu>[\d\.，,]+(十|百|千|万|十万|百万|千万|亿)?)|(?P<cn>(' + '|'.join(
    cnn) + ')+))'
pattern_str = '((?P<nu>[\d\.，,]+(十|百|千|万|十万|百万|千万|亿)?)|(?P<cn>(' + '|'.join(
    cnn) + ')+))'

nlp_logger = logger
tn = TimeNormalizer()
transfer_dic = {
    '【管理委员会会议纪要】': {
        '文件标题': '【管理委员会会议纪要】标题'
    },
    '【年度采购实施方案】': {
        '文件标题': '【年度采购实施方案】标题'
    },
    '【预算管理委员会会议纪要】': {
        '文件标题': '【预算管理委员会会议纪要】标题'
    },
    '【预算批复文件】': {
        '文件标题': '【预算批复文件】标题'
    },
    '【档案-封面】': {
        '项目名称': '【档案-封面】项目名称',
        '项目编号': '【档案-封面】项目编号',
        "采购方式": "【档案-封面】采购方式"
    },
    '【招标授权委托书】': {
        '委托有效期开始时间': '【招标授权委托书】授权期限开始时间',
        "项目名称": "【招标授权委托书】项目名称"
    },
    '【询价采购文件－封面】': {
        "文件签发日期": "【询价采购文件－封面】日期",
        "项目名称": "【询价采购文件－封面】项目名称"
    },
    '【询价采购文件】': {
        "供应商资质": "【询价采购文件】供应商资质"
    },
    '【合同模板】': {
        '合同名称': '【合同模板】合同名称',
        '验收方式': '【合同模板】验收方式',
        '付款条件描述': '【合同模板】付款条件描述',
        "付款比例": "【合同模板】付款比例"
    },
    '【廉洁合同模板】': {
        '合同名称': '【廉洁合同模板】合同名称'
    },
    '【供应商资格审查表】': {
        '项目名称': '【供应商资格审查表】项目名称',
        "文件签发日期": "【供应商资格审查表】时间"
    },
    '【【非招标项目谈判单位排查记录表】': {
        '项目名称': '【非招标项目谈判单位排查记录表】项目名称',
        '项目编号': '【非招标项目谈判单位排查记录表】项目编号'
    },
    '【成交供应商通知书】': {
        '项目名称': '【成交供应商通知书】项目名称',
        '文件签发日期': '【成交供应商通知书】发出日期',
        '落款名称': '【成交供应商通知书】落款名称',
        '成交供应商': '【成交供应商通知书】成交供应商'
    },
    '【合同签订授权委托书】': {
        '委托有效期开始时间': '【合同签订授权委托书】委托有效期开始时间',
        '文件签发日期': '【合同签订授权委托书】签订日期',
        '项目名称': '【合同签订授权委托书】项目名称',
        '成交供应商': '【合同签订授权委托书】成交供应商'
    },
    '【乙方授权委托书】': {
        '委托有效期开始时间': '【乙方授权委托书】委托有效期',
        '项目名称': '【乙方授权委托书】项目名称'
    },
    '【合同正本-封面】': {
        '合同名称': '【合同正本-封面】合同名称'
    },
    '【合同正本】': {
        '总金额': '【合同正本】总金额',
        '合同价格条款': '【合同正本】合同价格条款',
        '付款比例': '【合同正本】付款比例',
        '付款条件描述': '【合同正本】付款条件描述',
        '质保金': '【合同正本】质保金',
        '合同签订时间': '【合同正本】签订时间',
        '乙方违约金条款': '【合同正本】乙方违约金',
        '验收方式': '【合同正本】验收方式'
    },
    '【廉洁合同正本】': {
        '合同名称': '【廉洁合同正本】合同名称'
    },
}

content_key = {
    '【询价采购文件】': [{
        "行业黑名单": ["黑名单"]
    }, {
        "行贿数额": ["行贿数额在500万"]
    }],
    '【合同模板】': [{
        "转包分包": ["转包", "分包"]
    }, {
        "肢解": ["肢解"]
    }],
    '【廉洁合同模板】': [{
        "行业黑名单": ["黑名单"]
    }, {
        "行贿行为": ["行贿"]
    }],
    '【合同正本】': [{
        "转包分包": ["转包", "分包"]
    }, {
        "肢解": ["肢解"]
    }],
    '【廉洁合同正本】': [{
        "行业黑名单": ["黑名单"]
    }, {
        "行贿行为": ["行贿"]
    }],
}


def multi_position(text: str, sub: str) -> List[int]:
    """
    找到字符串中该sub_str的所有起始位置
    :param text:
    :param sub:
    :return:
    """
    res = []
    count = 0
    while True:
        pos = text.find(sub)
        if pos == -1:
            break
        else:
            res += [pos + count]
            count += pos + 1
            text = text[pos + 1:]

    return res


def get_line_text_from_line_info(line_info, start, end):
    for page, line_list in line_info.items():
        for line_map in line_list:
            if line_map['start'] <= start <= line_map['end']:
                return (line_map['txt'], line_map['start'], line_map['end'])

    return False


def get_num_info(id):
    try:
        headers = {'Content-Type': 'application/json'}
        res = requests.post(url=CONFIG_PARAM['SPLIT']['service'],
                            headers=headers,
                            json={"id": str(id)})
        c = res.content.decode('utf-8')
        c = json.loads(c)
        print(c)
        return c['data']['page_split_result']
    except Exception as e:
        nlp_logger.info(str(traceback.print_exc()))


def add_entity2res(entity_type, entity_value, doc_start, doc_end,
                   review_results, page_num, num_word):
    for page, num in page_num.items():
        if num[0] <= doc_start <= num[1]:
            if num_word[page] in transfer_dic.keys():
                # print(num_word[page])
                entity = transfer_dic[num_word[page]]
                # print(entity)
                # print(entity_type)
                if entity_type in entity.keys():
                    if entity[entity_type] == "【合同正本】总金额":
                        if cn2ara(entity_value):
                            ara = cn2ara(entity_value)
                        else:
                            ara = -1
                        review_results.append({
                            "name": entity[entity_type],  # "字段名称",
                            "match": entity_value,  # "匹配的原文内容",
                            "value": ara,
                            "is_left": 0,
                            "is_right": 1,
                            "start": doc_start,
                            "end": doc_end - 1
                        })
                    else:
                        review_results.append({
                            "name": entity[entity_type],  # "字段名称",
                            "match": entity_value,  # "匹配的原文内容",
                            "value": entity_value,
                            "is_left": 0,
                            "is_right": 1,
                            "start": doc_start,
                            "end": doc_end - 1
                        })
                else:
                    review_results.append({
                        "name": entity_type,  # "字段名称",
                        "match": entity_value,  # "匹配的原文内容",
                        "value": entity_value,
                        "is_left": 0,
                        "is_right": 1,
                        "start": doc_start,
                        "end": doc_end - 1
                    })
            else:
                review_results.append({
                    "name": entity_type,  # "字段名称",
                    "match": entity_value,  # "匹配的原文内容",
                    "value": entity_value,
                    "is_left": 0,
                    "is_right": 1,
                    "start": doc_start,
                    "end": doc_end - 1
                })


class ModelInference():
    def __init__(self, max_seq_length, checkpoint):
        self.ner_model = GeneralModelSpanPredict(max_seq_length=max_seq_length,
                                                 checkpoint=checkpoint)

    @time_cost
    def labels_inference(self, text: str, line_info, all_item, id):
        '''
                按句号分割,需要提升效率，现在是一句一句推理的
        '''
        review_results = []
        word_infos = all_item
        page_num = {}
        page_text = {}
        page_num[1] = [0, 1]
        temp_page = 1
        temp_text = ""
        temp_start = 0
        # print(len(word_infos))
        for i in range(1, len(word_infos)):
            if word_infos[i]['page'] == temp_page:
                temp = [page_num[temp_page][0], i]
                page_num[temp_page] = temp
            else:
                page_num[word_infos[i]['page']] = [i, i]
                temp_page = word_infos[i]['page']
        # print(page_num)
        temp_page = 1
        for i in range(len(word_infos)):
            if word_infos[i]['page'] == temp_page:
                temp_text += word_infos[i]['str']
            else:
                page_text[temp_page] = (temp_text, temp_start)
                temp_page = word_infos[i]['page']
                temp_start = i
                temp_text = word_infos[i]['str']

        temp_num_word = get_num_info(id)
        num_word = {}
        for k, v in temp_num_word.items():
            # if v != "未识别":
            num_word[int(k)] = v
        # assert len(page_text) == len(num_word)
        temp_file = ""
        temp_text = ""
        temp_start = 0
        file_texts = []
        temp_page = 1
        for page in page_text.keys():
            if num_word[page] == "【授权委托书】":
                with m_lock:
                    file_name = bert_predict({"text": [page_text[page][0]]})
                    # print("@"*100)
                    # print(page_text[page][0])
                    num_word[page] = file_name
                    # print(page)
                    # print(num_word[file_name])
                    # print("@"*100)
            if num_word[page] != temp_file and temp_file != "":
                file_texts.append((temp_file, temp_text, temp_start))
                temp_start = page_text[page][1]
                temp_text = ""
                temp_file = num_word[page]
            else:
                temp_file = num_word[page]
                temp_text += page_text[page][0]
        for file_text in file_texts:
            if file_text[0] in content_key.keys():
                for key_map in content_key[file_text[0]]:
                    exit_label = True
                    for k, v in key_map.items():
                        for vv in v:
                            if len(list(re.finditer(vv, file_text[1]))) == 0:
                                exit_label = False
                    if exit_label:
                        for k, v in key_map.items():
                            for vv in v:
                                for i in re.finditer(vv, file_text[1]):
                                    review_results.append({
                                        "name":
                                        file_text[0] + k,  # "字段名称",
                                        "match":
                                        i.group(),  # "匹配的原文内容",
                                        "value":
                                        i.group(),
                                        "is_left":
                                        0,
                                        "is_right":
                                        1,
                                        "start":
                                        i.span()[0] + file_text[2],
                                        "end":
                                        i.span()[1] + file_text[2] - 1
                                    })

            # print(file_text)

        sentences = re.split(r"([。])", text)
        sentences.append("")
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        sentences_len = [len(x) for x in sentences]
        for sentence_idx, sentence in enumerate(sentences):
            res_ = self.ner_model.predict(sentence)
            # print(res_)

            for res in res_:
                for x in res['entities']:
                    start = int(x[1])
                    end = int(x[2]) + 1
                    ner_obj = sentence[start:end]
                    doc_start = sum(sentences_len[:sentence_idx]) + start
                    doc_end = sum(sentences_len[:sentence_idx]) + end
                    assert sentence[start:end] == text[doc_start:doc_end]
                    entity_type = x[0]
                    entity_value = ner_obj
                    add_entity2res(entity_type, entity_value, doc_start,
                                   doc_end, review_results, page_num, num_word)

        return review_results