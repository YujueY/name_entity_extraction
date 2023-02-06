# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/11/17 5:49 下午 
@Author : azun
@File : time_pattern.py 
'''

import regex as re

cn_num = '一二三四五六七八九十零〇○o'

cn_amount = ['〇', '○', 'o', '一', '二', '三', '四', '五', '六', '七', '八', '九', '零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌',
             '玖', '貮', '两', '十', '拾', '百', '佰', '千', '仟', '万', '萬', '亿', '億', '兆']

# time_extraction_pattern = f'((\d|\.|/){{1,4}}|[{cn_num}]{{1,4}})?(年)?' \
#                           f'((\d|\.|/){{1,4}}|[{cn_num}]{{1,4}})?(月)' \
#                           f'((\d|\.|/){{1,4}}|[{cn_num}]{{1,4}})?(日)?(上午|下午|早上|晚上)?' \
#                           f'(((\d|\.|/){{1,2}}|[{cn_num}]{{1,2}})?(\(时/点\)|点|时|:|：))?' \
#                           f'(((\d|\.|/){{1,2}}|[{cn_num}]{{1,2}})?(分)?)?'
time_extraction_pattern = f'((\d|\.|/|{"|".join(list(cn_num))}){{1,4}})?(年)?' \
                          f'((\d|\.|/|{"|".join(list(cn_num))}){{1,4}})?(月)' \
                          f'((\d|\.|/|{"|".join(list(cn_num))}){{1,4}})?(日)?(上午|下午|早上|晚上)?' \
                          f'(((\d|\.|/|{"|".join(list(cn_num))}){{1,2}})?(\(时/点\)|点|时|:|：))?' \
                          f'(((\d|\.|/|{"|".join(list(cn_num))}){{1,2}})?(分)?)?'
# print(re.search(time_extraction_pattern, '日期：二0二0年四月').group(0))

ddl_drop_pattern = '(保证金递交|澄清|答疑|修改)'
provider_drop_pattern = '(保证金递交|澄清|答疑|修改)'
drop_Purchase_quantity_pattern = ('式')
drop_Number_of_service_providers_pattern = ('测评师')
drop_project_name = ('^\d$')
purchasing_content_service_pattern = '招标内容'
budget_extract_pattern = '(预算价|控制价|拦标价|合同价款)(共计|总计|金额|总额|为|人民币|.){0,6}?((?P<nu>[\d\.，,]+(十|百|千|万|十万|百万|千万|亿)?)|(?P<cn>(' + '|'.join(
    cn_amount) + ')+))(?P<unit>元)((大写|小写|.){0,7}?((?P<nu2>[\d\.，,]+(十|百|千|万|十万|百万|千万|亿)?)|(?P<cn2>(' + '|'.join(
    cn_amount) + ')+))(?P<unit2>元))?'

Bid_bond_extract_pattern = '(投标保证金|报价保证金)(保证金|共计|总计|总额|金额|为|人民币|.){0,6}?((?P<nu>[\d\.，,]+(十|百|千|万|十万|百万|千万|亿)?)|(?P<cn>(' + '|'.join(
    cn_amount) + ')+))(?P<unit>元)((大写|小写|.){0,7}?((?P<nu2>[\d\.，,]+(十|百|千|万|十万|百万|千万|亿)?)|(?P<cn2>(' + '|'.join(
    cn_amount) + ')+))(?P<unit2>元))?'

Warranty_Deposit_pattern = '(质量保证金|质保金)(保证金|共计|总计|总额|金额|为|人民币|.){0,6}?((?P<nu>[\d\.，,]+(十|百|千|万|十万|百万|千万|亿)?)|(?P<cn>(' + '|'.join(
    cn_amount) + ')+))(?P<unit>元)((大写|小写|.){0,7}?((?P<nu2>[\d\.，,]+(十|百|千|万|十万|百万|千万|亿)?)|(?P<cn2>(' + '|'.join(
    cn_amount) + ')+))(?P<unit2>元))?'

li = [
    '止时间及开标时开标时间：2020年5月日上午14:00(北京时间)。',
    '年月日年月日年月日', '2011年',
    '2020.1.1',
    '2020.1.',
    '2020..',
    '2020.1.1 12:30',
    '2020/1/1',
    '2020/01/0112:30',
    '2020年01月',
    '2020年1月1日',
    '2020年1月1日12:30',
    '2020年1月1日12(时/点)',
    '2020年1月1日12(时/点)30分',
    '二零二零年一月一日',
    '二〇二〇年一月一日',
    '二〇二〇年1月1日',
    '3个月',
    '二〇二〇年七月',
    '2020年8月3日16时00分',
    '2020年月日09:309'
]

# li=[
#     '2020年5月日14：00之前'
#
# ]
amount_l = [
    '★1、本次招标项目的投标保证金金额为：7000元。投标人的投标保证金应在2020年11月'
]
amount_pattern_map = {
    'Budget': budget_extract_pattern,
    'Bid_bond': Bid_bond_extract_pattern,
    'Warranty_Deposit': Warranty_Deposit_pattern,
}

if __name__ == '__main__':
    for x in li:
        print('x' * 80)
        print(x)
        search_obj = re.search(time_extraction_pattern, x)
        if search_obj:
            print(search_obj.group(0))
            print(search_obj.span())
    for text in amount_l:
        for entity_type, entity_type_pattern in amount_pattern_map.items():
            fiter = re.finditer(entity_type_pattern, text)
            # if fiter:
            #     print('=' * 80)
            #     print(k)
            #     print('=' * 80)
            for x in fiter:
                print('=' * 80)
                print(entity_type)
                print('=' * 80)
                x1, x2 = x.span()
                num = x['nu']
                cn = x['cn']
                unit = x['unit']
                print(num, '---', cn, '---', unit)
                l1 = x['l1']

                l2 = x['l2']

                l = 0
                l = len(l1)

                r = num if num else cn
                unit = unit if unit else ''
                entity_value = r + unit
                print('正则text:' + ''.join(text[x1 - 10:x2 + 20]))
                print('正则res1:' + entity_value)

                doc_start, doc_end = x.span()
                print(doc_start, doc_end)
                num = x['nu2']
                cn = x['cn2']
                unit = x['unit2']

                if num or cn:
                    print(num, '---', cn, '---', unit)
                    r = num if num else cn
                    unit = unit if unit else ''
                    entity_value = r + unit
                    print('res2:' + entity_value)
                    doc_start, doc_end = x.span()

        # print(re.search(ddl_drop_pattern,'招标文件答疑，澄清时间'))
        #
        # patternObj = re.search(budget_extract_pattern, '项目控制价为89879.877元jkhhkjhjkkjhkjjk控制价为89879.877元')
        # print(re.findall(budget_extract_pattern, '项目控制价为89879.877元jkhhkjhjkkjhkjjk控制价为89879.877元'))
        #
        # print(budget_extract_pattern)
        # fiter=re.finditer(budget_extract_pattern,'项目控制价为89879.877jkhhkjhjkkjhkjjk控制价为89879.877元')
        # fiter=re.finditer(budget_extract_pattern,'本合同价款共计人民币总额大写：贰拾伍万元整，(小写：)250000.00元')
        # # fiter=re.finditer(budget_extract_pattern,'预算价贰拾伍万元整')
        #
        #
        # for x in fiter:
        #     print('*'*30)
        #     print(list(range(x.span()[0],x.span()[1])))
        #     num = x['nu']
        #     cn = x['cn']
        #     unit = x['unit']
        #     print('num:',num)
        #     print('cn:',cn)
        #     print('unit:',unit)
        #
        #     r = num if num else cn
        #     unit =unit if unit else ''
        #     print(r+unit)
        #
        #     print('*' * 30)
        #     print(list(range(x.span()[0], x.span()[1])))
        #     num = x['nu2']
        #     cn = x['cn2']
        #     unit = x['unit2']
        #     print('num:', num)
        #     print('cn:', cn)
        #     print('unit:', unit)
        #
        #     r = num if num else cn
        #     unit = unit if unit else ''
        #     print(r + unit)
