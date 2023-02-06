# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/9/7 10:33 上午 
@Author : azun
@File : review.py 
'''
from controller.chinese_numerals2Arabic_numerals import cn2ara
from typing import *
from controller import NER_LABEL_TYPE, message, amount_label, time_label, exits_check_list, tips_check_list, \
    uniform_check_list, key_words
from controller.time_Norm.TimeNormalizer import TimeNormalizer
import traceback
import json
from prettyprinter import cpprint

tn = TimeNormalizer()


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


def uniformity_check(ner_label_dic: NER_LABEL_TYPE):
    '''
    判断上下文的一致性
    同一个实体，在上下文中检测出多个，判断是否严格相等
    :param ner_label_dic:
    :return:
    '''

    for entity_class, entity_info_list in ner_label_dic.items():
        _ = []

        for entity_info in entity_info_list:
            entity = entity_info['entity']
            _.append(entity)

        if 1 != len(set(_)):
            for entity_info in entity_info_list:
                entity_info['message'] = message[entity_class]
                entity_info['risk_level'] = 3


def exist_check(ner_label_dic: NER_LABEL_TYPE):
    '''
    判断是否存在
    :param ner_label_dic:
    :return:
    '''

    #############################################################################
    for entity_class, entity_info_list in ner_label_dic.items():
        if entity_class in exits_check_list and (not entity_info_list):
            entity_info_list.append(
                {
                    'entity': entity_class,
                    'start': -1,
                    'end': -1,
                    'risk_level': 3,
                    'message': message['Audit_reduction_amount']
                }

            )
    #############################################################################
    #############################################################################


def amount_check(ner_label_dic: NER_LABEL_TYPE, text: str):
    '''
    数值判断
    :param ner_label_dic:
    :param text:
    :return:
    '''
    if ner_label_dic['Budget'] and ner_label_dic['Bid_bond']:
        budget_dig = ner_label_dic['Budget'][0]['entity']

        for bid_bond_info in ner_label_dic['Bid_bond']:

            try:
                budget_dig = cn2ara(budget_dig)
                bid_bond_dig = cn2ara(bid_bond_info['entity'])

                if bid_bond_dig > budget_dig * 0.02:
                    # print('budget_dig', budget_dig)
                    # print('bid_bond_dig', bid_bond_dig)
                    bid_bond_info['message'] = message['Bid_bond']
                    bid_bond_info['risk_level'] = 3
            except:
                bid_bond_info['message'] = message['Bid_bond']
                bid_bond_info['risk_level'] = 3
                print(f"budget_dig{budget_dig}", flush=True)
                print(f"bid_bond_info['entity']:{bid_bond_info['entity']}", flush=True)
                traceback.print_exc()

    elif not ner_label_dic['Budget'] and ner_label_dic['Bid_bond']:
        for bid_bond_info in ner_label_dic['Bid_bond']:
            bid_bond_info['message'] = message['Bid_bond']
            bid_bond_info['risk_level'] = 3

    if ner_label_dic['Budget'] and ner_label_dic['Warranty_Deposit']:
        budget_dig = ner_label_dic['Budget'][0]['entity']

        for warranty_Deposit_info in ner_label_dic['Warranty_Deposit']:
            try:
                budget_dig = cn2ara(budget_dig)
                warranty_Deposit_dig = cn2ara(warranty_Deposit_info['entity'])
                if warranty_Deposit_dig < budget_dig * 0.05:
                    warranty_Deposit_info['message'] = message['Warranty_Deposit']
                    warranty_Deposit_info['risk_level'] = 3
            except:
                warranty_Deposit_info['message'] = message['Warranty_Deposit']
                warranty_Deposit_info['risk_level'] = 3
                print(f"budget_dig{budget_dig}", flush=True)
                print(f"warranty_Deposit_info['entity']:{warranty_Deposit_info['entity']}", flush=True)
                traceback.print_exc()

    elif not ner_label_dic['Budget'] and ner_label_dic['Warranty_Deposit']:
        for warranty_Deposit_info in ner_label_dic['Warranty_Deposit']:
            warranty_Deposit_info['message'] = message['Warranty_Deposit']
            warranty_Deposit_info['risk_level'] = 3

    if ner_label_dic['Budget']:
        budget_dig = ner_label_dic['Budget'][0]['entity']
        try:
            budget_dig = cn2ara(budget_dig)
            if budget_dig >= 5000000:
                if '回标分析' not in text:
                    # 如果没有，我怎么传message过去
                    ner_label_dic['Back_bid_analysis'].append({
                        'entity': "回标分析",
                        'start': -1,
                        'end': -1,
                        'risk_level': 3,
                        'message': message['Back_bid_analysis']
                    })

        except Exception as e:
            traceback.print_exc()
            # print(e)


def media_check(ner_label_dic: NER_LABEL_TYPE):
    '''
    公告发布媒体一定包含(中国采购与招标网、山东省采购与招标网、中国招标投标公共服务平台)

    :param ner_label_dic:
    :return:
    '''
    if ner_label_dic['Announcement_media']:
        for entity_info in ner_label_dic['Announcement_media']:
            # print('公告发布媒体一定包含')
            # print(ner_label_dic['Overall']['Announcement_media']['items'])
            if not ('中国采购与招标网' in entity_info['entity'] and '山东省采购与招标网' in entity_info['entity'] and '中国招标投标公共服务平台' in
                    entity_info['entity']):
                entity_info['message'] = message['Announcement_media']
                entity_info['risk_level'] = 3


# def audit_reduction_amount_check(ner_label_dic):
#     if ner_label_dic['Audit_reduction_amount']:
#         for entity_info in ner_label_dic['Audit_reduction_amount']:
#             entity_info['message'] = message['Audit_reduction_amount']


def tips_check(ner_label_dic: NER_LABEL_TYPE):
    '''
    小提示，只要出现某个关键词，就加上提示语
    :param ner_label_dic:
    :return:
    '''

    for tips_entity in tips_check_list:
        if ner_label_dic[tips_entity]:
            for entity_info in ner_label_dic[tips_entity]:
                entity_info['message'] = message['Purchasing_content_service']
                entity_info['risk_level'] = 0
    # if not ner_label_dic['Overall']['Tender_time']['items']:
    #     ner_label_dic['Overall']['Tender_time']['message'] = [message['Tender_time']]
    # pass


def key_word_check(text, ner_label_dic):
    # 先取消ner的结果，因为是改用了关键字

    ner_label_dic['Working_experience'] = []

    for key_word, key_word_zh in key_words.items():
        for _idx in multi_position(text, key_word_zh):
            ner_label_dic[key_word].append(
                {
                    'entity': key_word_zh,
                    'start': _idx,
                    'end': _idx + len(key_word_zh) - 1,
                    'risk_level': 0,
                    'message': message['Audit_reduction_amount']
                }
            )
