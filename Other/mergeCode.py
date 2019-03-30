# -*- coding: utf-8 -*-
#-------------------------------------------------
#   File Name：     mergeCode
#   Author :        fengge
#   date：          2019/3/28
#   Description :   07,08 细分 ，其他学科不细分
#-------------------------------------------------
from  Util.db_util import dbutil
def merge():
    dbs = dbutil("local")
    sql="select INSTITUTION_ID from `es_relation_in_dis` GROUP BY INSTITUTION_ID"
    INSTITUTION=dbs.getDics(sql)
    sql="select INSTITUTION_ID,DISCIPLINE_CODE from es_relation_in_dis where INSTITUTION_ID=%s"
    for i in INSTITUTION:
        codes=dbs.getDics(sql,(i['INSTITUTION_ID'],))
        name=[code['DISCIPLINE_CODE'] for code in codes]
        for prefix in name:
            if len(prefix)>=4 and (prefix.startswith('07') or prefix.startswith('08')):
                new_code=prefix[0:4]
            elif len(prefix)>=2:
                new_code=prefix[0:2]
            else:
                print("error:",prefix)
                continue
            item={}
            item['table']="es_relation_in_dis2"
            item['params'] = {"INSTITUTION_ID":i['INSTITUTION_ID'],"DISCIPLINE_CODE":new_code}
            try:
                dbs.insertItem(item)
            except:
                pass
import re


def teacher_paper_code():
    sql="select a.*,b.name  from (SELECT INSTITUTION_ID, GROUP_CONCAT(DISCIPLINE_CODE ORDER BY DISCIPLINE_CODE ASC SEPARATOR ',') AS DISCIPLINE_CODE FROM es_relation_in_dis2 GROUP BY INSTITUTION_ID) a left join es_discipline b on a.DISCIPLINE_CODE=b.code"
    dbs = dbutil("local")
    INSTITUTION = dbs.getDics(sql)
    sql="select p.id,p.author_id,p.name title,p.abstract,p.keyword,t.`NAME`  from es_paper p join es_teacher t on p.author_id=t.id and t.INSTITUTION_ID=%s"
    for inde,i in enumerate(INSTITUTION):
        print(inde,i)
        paper=dbs.getDics(sql,(i['INSTITUTION_ID'],))
        for p in paper:
            p["DISCIPLINE_CODE"]=i["DISCIPLINE_CODE"]
            p["DISCIPLINE_NAME"] = i["name"]
            p['abstract']= re.sub('\s', '',p['abstract'])
            item={}
            item['table']="paper_teacher_code"
            item['params'] = p
            # print(p)
            dbs.insertItem(item)


if __name__=="__main__":
    # merge()
    teacher_paper_code()
    pass