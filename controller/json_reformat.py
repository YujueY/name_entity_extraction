# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/9/8 3:29 下午 
@Author : azun
@File : json_reformat.py 
'''
from controller import NER_LABEL_TYPE
from controller import doc_class_entity_info,label_zh_dic
import traceback
from prettyprinter import cpprint,pformat
from utils.logClass import logger
from controller.time_Norm.TimeNormalizer import TimeNormalizer

# print(logger)
# logger.info("asdhadasdada")
nlp_logger = logger
tn = TimeNormalizer()
import traceback

def json_reformat(doc_class,json_res:NER_LABEL_TYPE):
    review_results = []
    score = 100

    pass_item_count = 0
    total_item_count = 0
    risk_item_count_1 = 0
    risk_item_count_2 = 0
    risk_item_count_3 = 0



    for entity_type, entity_info in json_res.items():

        # 不在该类内的实体，跳过
        if entity_type not in doc_class_entity_info[doc_class]:
            continue

        has_risk_level_3=False
        for entity in entity_info:
            review_results.append(
                {
                    "name": label_zh_dic[entity_type],  # "字段名称",
                    "match": entity['entity'],  # "匹配的原文内容",
                    "level": entity['risk_level'],  # 风险等级 1/2/3
                    "review_result": {
                        "law": '',  # 法律条文",
                        "desc": entity['message'],  # "风险点说明"
                    },
                    "is_left": 0,
                    "is_right": 1,
                    "start": entity['start'],
                    "end": entity['end']
                }

            )
            if entity['risk_level']==0:
                pass_item_count+=1
            if entity['risk_level']==1:
                risk_item_count_1+=1
            if entity['risk_level']==2:
                risk_item_count_2+=1
            elif entity['risk_level']==3:
                risk_item_count_3+=1
                has_risk_level_3=True
        if has_risk_level_3:
            score-=20




    score_result = {
        "score": max(score,0),
        'pass_item_count': pass_item_count,
        'total_item_count': len(review_results),
        'risk_item_count_1': risk_item_count_1,
        'risk_item_count_2': risk_item_count_2,
        'risk_item_count_3': risk_item_count_3,
    }




    return review_results, score_result

# for a,b in label_zh_dic.items():
#     print(b)
