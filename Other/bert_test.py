# -*- coding: utf-8 -*-
#-------------------------------------------------
#   File Name：     bert_test
#   Author :        fengge
#   date：          2019/3/29
#   Description :   测试bert
#-------------------------------------------------

from config import data_dir
from bert_serving.client import BertClient
from sklearn.manifold import TSNE
from  Util.db_util import dbutil
def getBertVec():
    dbs = dbutil("local")
    bc = BertClient()  # ip address of the GPU machine
    sql="select * from paper_teacher_code limit %s,%s"
    size=200
    start=0
    file=open(data_dir+"/Data/paperVec.txt",'w',encoding='utf8')
    while True:
        print(start)
        paper=dbs.getDics(sql,(start,size))
        if len(paper)==0:
            break
        start+=size
        text=[p['title']+'。'+p['abstract']+'。'+p['keyword'] for p in paper]
    # doing encoding in one-shot
        vec = bc.encode(text)
        for i,p in enumerate(paper):
            p["vec"]=[n for n in vec[i]]
            file.write(str(p)+'\n')
    file.close()
def desVec():
    files = open(data_dir + "/Data/paperVec.txt", 'r', encoding='utf8')

    start=0
    ndpp=[]
    ids=[]
    print("load data")
    while True:
        line = files.readline()
        if not line:
            break

        dict=eval(line)
        ndpp.append(dict['vec'])
        ids.append(dict['id'])
    files.close()
    print("desc data")
    tsne = TSNE(perplexity=200)
    X = tsne.fit_transform(ndpp)
    files = open(data_dir + "/Data/paperDesVec.txt", 'w', encoding='utf8')
    print("save data")
    for i,id in enumerate(ids):
        item={'id':id,"vec":list(X[i])}
        files.write(str(item)+'\n')
    files.close()




if __name__ == "__main__":
    desVec()
    pass
