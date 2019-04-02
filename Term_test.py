# -*- coding: utf-8 -*-
#-------------------------------------------------
#   File Name：     Term_test
#   Author :        fengge
#   date：          2019/3/27
#   Description :   术语抽取测试
#-------------------------------------------------
from copy import deepcopy
from Term_Extraction.LoadData import getData
from Term_Extraction.ExportUserDict import ExporUserDict
from Term_Extraction.FindRuleTerminology import getTermRule,fenci,txtToPickle
from  Util.db_util import dbutil
import pickle
if __name__=="__main__":
    # dbs = dbutil("local")
    # sql="select DISTINCT DISCIPLINE_CODE  from es_relation_in_dis2"
    # code=dbs.getDics(sql)
    # code_list=[(c['DISCIPLINE_CODE'],"%",c['DISCIPLINE_CODE'],"%","%",c['DISCIPLINE_CODE'],"%") for c in code]
    # sql="select * from paper_teacher_code where DISCIPLINE_CODE ='%s' or DISCIPLINE_CODE like '%s%s,%s' or DISCIPLINE_CODE like '%s,%s%s'"
    # paper=getData(dbs,code_list,sql,name_index=0)
    #
    # f = open('Data/paper', 'wb')
    # pickle.dump(paper, f)
    # f = open('Data/paper', 'rb')
    # paper=pickle.load(f)
    # word=set()
    # for co in paper:
    #     for p in paper[co]:
    #         w=p['keyword'].split(',')
    #         print(w)
    #         for t in w:
    #             word.add(t)
    # word=list(word)
    # ExporUserDict(word,infile="/Data/userdict.txt")
    # fenci(paper)

    txtToPickle()
    f = open('Data/paper2', 'rb')
    paper=pickle.load(f)
    # getTermRule(paper)

    pass


