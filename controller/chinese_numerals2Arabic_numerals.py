# -*- coding: utf-8 -*- 
'''
@version V1.0.0
@Time : 2021/5/6 1:55 下午 
@Author : azun
@File : chinese_numerals2Arabic_numerals.py 
'''
import regex as re
'''
中文汉字到阿拉伯汉字的转换
主要是金额方面

'''

CN_NUM = {
    u'〇': 0,
    u'一': 1,
    u'二': 2,
    u'三': 3,
    u'四': 4,
    u'五': 5,
    u'六': 6,
    u'七': 7,
    u'八': 8,
    u'九': 9,

    u'零': 0,
    u'壹': 1,
    u'贰': 2,
    u'叁': 3,
    u'肆': 4,
    u'伍': 5,
    u'陆': 6,
    u'柒': 7,
    u'捌': 8,
    u'玖': 9,
    u'貮': 2,
    u'两': 2,
}
CN_UNIT = {
    u'十': 10,
    u'拾': 10,
    u'百': 100,
    u'佰': 100,
    u'千': 1000,
    u'仟': 1000,
    u'万': 10000,
    u'萬': 10000,
    u'亿': 100000000,
    u'億': 100000000,
    u'兆': 1000000000000,
}


# print(list(CN_NUM.keys())+list(CN_UNIT.keys()))



def delete_extra_zero(n):
    '''删除小数点后多余的0'''
    if isinstance(n, int):
        return n
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        return n

def is_digital(text):
    try:
        float(text)
        return True
    except:
        return False

def cn2ara(cn):


    cn=str(cn)
    cn=re.sub('元.*$','',cn)
    x=is_digital(cn)
    if x:

        return delete_extra_zero(float(cn))
    pattern='(?P<num>[\d\.]+)(?P<cn>万|亿)'

    searchObj=re.search(pattern,cn)
    if searchObj:
        num_part=searchObj.group('num')
        cn_part=searchObj.group('cn')

        try:
            num_part=float(num_part)
            if cn_part=='万':
                return int(num_part*10000)
            elif cn_part=='亿':
                return int(num_part * 10000* 10000)
        except:
            pass

    try:
        int(cn)
        return int(cn)

    except:
        pass


    lcn = list(cn)
    unit = 0  # 当前的单位
    ldig = []  # 临时数组

    ret = 0
    tmp = 0

    if lcn[-2:]==['零', '十']:    # “一千零十”
        ret = 10
        lcn = lcn[:-2]

    while lcn:
        cndig = lcn.pop()
        if cndig in CN_UNIT:          # python2: CN_UNIT.has_key(cndig)
            unit = CN_UNIT.get(cndig)
            if unit == 10000:
                ldig.append('w')  # 标示万位
                unit = 1
            elif unit == 100000000:
                ldig.append('y')  # 标示亿位
                unit = 1
            elif unit == 1000000000000:  # 标示兆位
                ldig.append('z')
                unit = 1
            continue
        else:
            dig = CN_NUM.get(cndig)
            if dig == None:
                return False
            else:
                if unit:
                    dig = dig * unit

                    if len(ldig)==1 and isinstance(ldig[0],int) and ldig[0]<10:   # “七百五”“七千五”的“五”不是个位
                        ldig[0] = ldig[0]*unit//10
                    if len(ldig)==2 and isinstance(ldig[0],int) and ldig[0]<10:   # “七亿五”“七万五”的“五”不是个位
                        if ldig[-1]=='w':
                            ldig[0] = ldig[0] * 1000
                        elif ldig[-1]=='y':
                            ldig[0] = ldig[0] * 10000000
                        elif ldig[-1]=='z':
                            ldig[0] = ldig[0] * 100000000000
                    unit = 0

                ldig.append(dig)

    if unit == 10:  # 处理10-19的数字
        ldig.append(10)

    while ldig:
        x = ldig.pop()
        if x == 'w':
            tmp *= 10000
            ret += tmp
            tmp = 0
        elif x == 'y':
            tmp *= 100000000
            ret += tmp
            tmp = 0
        elif x == 'z':
            tmp *= 1000000000000
            ret += tmp
            tmp = 0
        else:
            tmp += x
    ret += tmp
    return int(ret)


print(cn2ara('一万两千三百四十元'))