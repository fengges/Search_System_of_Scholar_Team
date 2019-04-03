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
    f = open('Data/paper', 'rb')
    paper=pickle.load(f)
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
    code=["0805", "0812", "03", "05", "0808", "0802", "0807", "0830", "12", "0835", "10", "0703", "02", "0811", "0710", "0810", "0701", "0702", "0814", "0823", "06", "09", "0832", "0705", "0809", "0815", "13", "04", "0831", "0709", "0714", "0804", "0817", "0822", "0833", "0801", "01", "0706", "0707", "0827", "0825", "0806", "0803", "0712", "0704", "0713", "0824", "0816", "0834", "0711"]
    code_list=code[0:10]
    txtToPickle(code_list=code_list)
    f = open('Data/paper2', 'rb')
    paper=pickle.load(f)
    getTermRule(paper)

    pass


