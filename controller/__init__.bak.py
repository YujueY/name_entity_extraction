# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/9/7 3:19 下午 
@Author : azun
@File : __init__.py 
'''
from typing import *

NER_LABEL_TYPE = Dict[
    # str,Dict[str,List]
    str, List
]

amount_label = [
    'Budget', 'Bid_bond', 'Warranty_Deposit'
]
time_label = [
    # 'Deadline', 'Tender_time', 'Service_period', 'Delivery_time', 'Construction_period'
]

message = {
    'Project_name': '包含《计划表》xls中的项目名称；有且在文章中出现多次，上下文保持一致；',
    'Budget': '小于等于《计划表》中的项目金额；在文章中出现多次，上下文保持一致；',
    'Deadline': '必须要有；在文章中出现多次，上下文保持一致；',
    'Bid_bond': '必须要有；小于等于控制价（预算价）的2%；两个金额抓出来做比较。',
    'Warranty_Deposit': '不一定要有；不低于控制价（预算价）5%。有且在文章中出现多次，上下文保持一致；',
    'Tender_time': '一定要有，抽取出，提示语“不得早于招标代理机构选择日期”',
    'Scoring_standard': '一定要有',
    'Qualifications_of_bidders': '一定要有',
    'Contract_standard_clause': '一定要有',
    'Announcement_media': '一定包含这三家：中国招标投标公共服务平台、中国采购与招标网、山东省采购与招标网。',
    'Purchasing_content_service': '（服务类采购内容需包含1、一定要有服务期或者工期；2、一定要有金额；',
    'Number_of_service_providers': '一定要有，一定存在服务商（中标人、供应商）（候选人）的数量说明；在文章中出现多次，上下文保持一致；',
    'Service_period': '一定要有，采购类合同需要明确服务年限；在文章中出现多次，上下文保持一致；',
    'Purchase_quantity': '一定要有，物资类采购合同需明确采购数量',
    'Delivery_time': '一定要有,采购类合同需要明确供货期；在文章中出现多次，上下文保持一致；',
    'Construction_period': '一定要有,工程类需要明确施工期（工期）；在文章中出现多次，上下文保持一致；',
    'Audit_reduction_amount': '需要有“审减金额超过预算价的10%或地方规定最低比例时，超过部分审计费用由施工单位承担”。或者“审减金额超过预算价的5%时，超过部分审计费用由施工单位承担”',
    'Working_experience': '信息化、工程类公告请人工检查项目组人员的从业经验，执业资格是否合格。',
    'Project_period': '信息化、工程类公告请人工检查项目工期应符合国家、行业和地方建设行政管理部门相关规定',
    'Technical_requirements': '信息化项目需明确技术要求、技术指标。',
    'Back_bid_analysis': '工程类预算价（控制价、拦标价）在500万以上，需要有回标分析。（需求分析：预算价（控制价、拦标价）在500万以上，需要有“回标分析”这四个字）。',
}

uniform_check_list = [
        'Project_name',
        'Budget',
        'Deadline',
        'Warranty_Deposit',
        'Number_of_service_providers',
        'Service_period',
        'Delivery_time',
        'Construction_period']
ner_map = {
    'project_name': 'Project_name',
    'Budget': 'Budget',
    'Deadline': 'Deadline',
    'Bid_bond': 'Bid_bond',
    'Warranty_Deposit': 'Warranty_Deposit',
    'Tender_time': 'Tender_time',
    'Scoring_standard': 'Scoring_standard',
    'Qualifications_of_bidders': 'Qualifications_of_bidders',
    'Contract_standard_clause': 'Contract_standard_clause',
    'Announcement_media': 'Announcement_media',
    'Purchasing_content_service': 'Purchasing_content_service',
    'Number_of_service_providers': 'Number_of_service_providers',
    'Service_period': 'Service_period',
    'Purchase_quantity': 'Purchase_quantity',
    'Delivery_time': 'Delivery_time',
    'Construction_period': 'Construction_period',
    'Audit_reduction_amount': 'Audit_reduction_amount',
    'working_experience': 'Working_experience',
    'construction_period': 'Project_period',
    'Technical_requirements': 'Technical_requirements'

}
tips_check_list = ['Purchasing_content_service', 'Working_experience', 'Project_period', 'Technical_requirements']

exits_check_list = [
    'Project_name',
    'Deadline',
    'Bid_bond',
    'Tender_time',
    'Scoring_standard',
    'Qualifications_of_bidders',
    'Contract_standard_clause',
    'Number_of_service_providers',
    'Service_period',
    'Purchase_quantity',
    'Delivery_time',
    'Construction_period',
    'Audit_reduction_amount',
    'Announcement_media'

]

ner_label_dic = {
    'Overall':
        {
            'Project_name': {},
            'Budget': {},
            'Deadline': {},
            'Bid_bond': {},
            'Warranty_Deposit': {},
            'Tender_time': {},
            'Scoring_standard': {},
            'Qualifications_of_bidders': {},
            'Contract_standard_clause': {},
            'Announcement_media': {},
        },
    'Service': {
        'Purchasing_content_service': {},
        'Number_of_service_providers': {},
        'Service_period': {},
    },
    'Purchase': {
        'Purchase_quantity': {},
        'Delivery_time': {},
    },
    'Investment': {
        'Construction_period': {},
        'Audit_reduction_amount': {},
        'Back_bid_analysis': {},
        'Working_experience': {},
        "Project_period": {},
        'Technical_requirements': {}
    }
}

label_zh_dic = {
    'Project_name': '项目名称',
    'Budget': '预算价、控制价、拦标价',
    'Deadline': '截止日期、开标时间、截止时间、投标截止',
    'Bid_bond': '投标保证金',
    'Warranty_Deposit': '质保金（质量保证金）',
    'Tender_time': '标书时间',
    'Scoring_standard': '评标办法、评分标准',
    'Qualifications_of_bidders': '投标人资格、投标单位资格、合格的投标人',
    'Contract_standard_clause': '合同格式条款',
    'Announcement_media': '公告发布媒体',
    'Purchasing_content_service': '采购内容-服务类',
    'Number_of_service_providers': '服务商数量',
    'Service_period': '服务年限、服务期',
    'Purchase_quantity': '采购数量',
    'Delivery_time': '供货期（供货时间，交货时间）',
    'Construction_period': '施工期、工期',
    'Project_period': '项目工期',
    'Audit_reduction_amount': '审减金额',
    'Working_experience': '项目组人员的从业经验，执业资格。',
    'Technical_requirements': '技术要求、技术指标',
    'Back_bid_analysis': '回标分析',
}

ner_label_dic_templet={
    k2:[] for k1,v1 in ner_label_dic.items() for k2,v2 in v1.items()
}
# for k1, v1 in ner_label_dic_templet.items():
#     for k2, v2 in v1.items():
#         print(k2)
        # v1[k2] = {
        #     'items': [
        #     ],
        #     'message': []
        # }
# ner_label_class = {
#     k2: k1 for k1, v1 in ner_label_dic_templet.items() for k2, v2 in v1.items()
# }

doc_class_entity_info={

    'Overall': list(ner_label_dic['Overall'].keys()),
    'Service':list(ner_label_dic['Overall'].keys())+list(ner_label_dic['Service'].keys()),
    'Purchase':list(ner_label_dic['Overall'].keys())+list(ner_label_dic['Purchase'].keys()),
    'Investment':list(ner_label_dic['Overall'].keys())+list(ner_label_dic['Investment'].keys())


}

key_words={
    "Working_experience":'人员要求'
}