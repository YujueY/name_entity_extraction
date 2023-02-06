# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/9/7 11:42 下午
@Author : azun
@File : review.py 
'''
from controller.review_fn import uniformity_check, amount_check, exist_check, tips_check, media_check, message
from controller import NER_LABEL_TYPE
from typing import *


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


def review(ner_label_dic: NER_LABEL_TYPE, text: str):
    """
           直接判断全文是否有回标分析四个字
   """
    # if '回标分析' in text:
    #     for _idx in multi_position(text, '回标分析'):
    #         ner_label_dic['Investment']['Back_bid_analysis']['items'].append(
    #             {
    #                 'entity': '回标分析',
    #                 'start': _idx,
    #                 'end': _idx + 4
    #             }
    #         )

    Audit_reduction_amount_content1 = "审减金额超过预算价的10%或地方规定最低比例时，超过部分审计费用由施工单位承担"
    Audit_reduction_amount_content2 = "审减金额超过预算价的5%时，超过部分审计费用由施工单位承担"

    #  if 审减金额超过预算价的10%或地方规定最低比例时，超过部分审计费用由施工单位承担 exists
    if Audit_reduction_amount_content1 in text or Audit_reduction_amount_content2 in text:
        for _idx in multi_position(text, Audit_reduction_amount_content1):
            ner_label_dic['Audit_reduction_amount'].append(
                {
                    'entity': Audit_reduction_amount_content1,
                    'start': _idx,
                    'end': _idx + len(Audit_reduction_amount_content1) - 1,
                    'risk_level': 0,
                    'message': message['Audit_reduction_amount']
                }
                #
                # {
                #     'entity': Audit_reduction_amount_content1,
                #     'start': _idx,
                #     'end': _idx + len(Audit_reduction_amount_content2)
                # }
            )
        for _idx in multi_position(text, Audit_reduction_amount_content2):
            ner_label_dic['Audit_reduction_amount'].append(
                {
                    'entity': Audit_reduction_amount_content2,
                    'start': _idx,
                    'end': _idx + len(Audit_reduction_amount_content2) - 1,
                    'risk_level': 0,
                    'message': message['Audit_reduction_amount']
                }
            )

    uniformity_check(ner_label_dic)
    amount_check(ner_label_dic, text)
    tips_check(ner_label_dic)
    media_check(ner_label_dic)
    #############################################################################
    # 由于是否存在这个检测交给了后端，每个扣一分，我这里就不检测了
    # exist_check(ner_label_dic)
    #############################################################################

    # audit_reduction_amount_check(ner_label_dic)
