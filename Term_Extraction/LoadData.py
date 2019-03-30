# -*- coding: utf-8 -*-
#-------------------------------------------------
#   File Name：     LoadData
#   Author :        fengge
#   date：          2019/3/27
#   Description :   加载数据
#-------------------------------------------------
import tqdm
def getData(dbs,code_list,sql,name_index=None):
    '''
    通过条件加载数据
    Parameter:
        - dbs: 数据库链接
        - code_list: 不同领域代码
        - outfile:添加的词典位置
        - sql: 查询sql
        - name_index: 如何显示名字
    '''
    paper={}
    for code in code_list:
        tmp_sql=sql%code
        index=code[name_index] if name_index is not None else code
        paper[index]=dbs.getDics(tmp_sql)
        print(index,":",len(paper[index]))
    return paper