# -*- coding: utf-8 -*-
#-------------------------------------------------
#   File Name：     FindRuleTerminology
#   Author :        fengge
#   date：          2019/3/27
#   Description :   发现术语词性规则
#-------------------------------------------------
import jieba,os
from copy import deepcopy
import jieba.posseg as pseg
from config import data_dir,tmp_dir
from tqdm import tqdm
def txtToPickle(code_list=[],input=data_dir+"/Data/paper_fenci.txt",output=data_dir+"/Data/paper2"):
    '''
    Parameter:
        - code_list: 保留的编码
        - input:    输入的文件
        - output:  输出的文件
    '''
    paper2={}
    files=open(input,'r',encoding='utf8')
    while True:
        line=files.readline()
        if not line:
            break
        dict=eval(line)
        if code_list and dict["DISCIPLINE_CODE"] not in code_list:
            continue
        if dict["DISCIPLINE_CODE"] not in paper2:
            paper2[dict["DISCIPLINE_CODE"]]=[]
        paper2[dict["DISCIPLINE_CODE"]].append(dict)
    pickle.dump(paper2,open(output,'wb'))
def fenci(paper,userdict=data_dir+"/Data/user.txt",out=data_dir+"/Data/paper_fenci.txt"):
    '''
    分词
    Parameter:
        - paper: 每个领域的文章
        - userdict: 自定义词典的位置
        - out:  文件输出位置
    '''
    if os.path.exists(userdict):
        jieba.load_userdict(userdict)
    file=open(out,'w',encoding='utf8')
    for code in paper:
        # print('code:',code)
        for data in tqdm(paper[code]):
            t_seg = [(word, flag) for word, flag in pseg.cut(data["title"])]
            d_seg = [(word, flag) for word, flag in pseg.cut(data["abstract"])]
            item={'id':data["id"],"t_seg":t_seg,"d_seg":d_seg,"DISCIPLINE_CODE":code}
            file.write(str(item)+'\n')
    file.close()
def getTermRule(paper,stop_flag=None,n_gram=[2,6],max_step=10):
    '''
    统计术语词性组成规则
    Parameter:
        - paper: 每个领域的文章
        - stop_flag: 停用词词性
        - n_gram: 术语组成规则长度范围  n_gram[0]-n_gram[1]
        - max_step: 头尾最大间距
    '''
    # if os.path.exists(userdict):
    #     jieba.load_userdict(userdict)
    if not stop_flag:
        stop_flag = ['f', 's', 'r', 'm', 'z', 'q', 'u', 'ul', 'y', 'vg', 't', 'p', 'o', 'c',  'tg', 'i']
    all_rule={}
    for code in paper:
        # code="01"
        print('code:',code)
        print("len:",len(paper[code]))
        code_rule={}
        start=0
        # for data in tqdm(paper[code]):
        for data in paper[code]:
            start+=1
            if start%1000==0:
                print(start)
            # if start<31625:
            #     continue
            t_seg = data["t_seg"]
            d_seg = data["d_seg"]

            t_word = []
            t_flag = []
            d_word = []
            d_flag = []

            for word, flag in t_seg:
                t_word.append(word)
                t_flag.append(flag)
            for word, flag in d_seg:
                d_word.append(word)
                d_flag.append(flag)
            #  通过停用词词性 将摘要分割
            sentence=[[]]
            for i,d in enumerate(d_flag):
                if d in stop_flag:
                    sentence.append([])
                else:
                    sentence[-1].append(d_word[i])
            for sen in sentence:
                _,substr=getLCS(sen,t_word)
                for sub in substr:
                    tmp=[]
                    getSub(sub,0,[],tmp)
                    for t in tmp:
                        if len(t)>=n_gram[0] and len(t)<=n_gram[1] and t[-1][-1]-t[0][-1]<max_step:
                            rule="_".join([t_flag[ws[-1]]  for ws in t])
                            long_word="_".join([ws[0]  for ws in t])
                            if rule not in code_rule:
                                code_rule[rule]={"all":0}
                            if long_word not in code_rule[rule]:
                                code_rule[rule][long_word]=0
                            code_rule[rule][long_word]+=1
                            code_rule[rule]["all"]+=1
        file_dict={i:open(tmp_dir+"/Data_tmp/rule_"+code+"_"+str(i)+'.txt','w',encoding='utf8')   for i in range(n_gram[0],n_gram[1]+1)}
        temp=sorted(code_rule.items(),key=lambda x:x[1]['all'],reverse=True)
        _size=len(temp)//10+1
        temp=temp[0:_size]
        for t in temp:
            size=len(t[0].split("_"))
            file_dict[size].write(t[0]+":"+str(t[1]))
        for i in file_dict:
            file_dict[i].close()
    #     for tmp in code_rule:
    #         if tmp not in all_rule:
    #             all_rule[tmp]=0
    #         all_rule[tmp]+=code_rule[tmp]["all"]
    #
    # file_dict2={i:open(data_dir+"/Data_tmp/rule_"+str(i)+'.txt','w',encoding='utf8')   for i in range(n_gram[0],n_gram[1]+1)}
    # temp=sorted(all_rule.items(),key=lambda x:x[1],reverse=True)
    # for t in temp:
    #     size=len(t[0].split("_"))
    #     file_dict2[size].write(t[0]+":"+str(t[1]))

#  找到匹配串，的子串
def getSub(test, i, res,r):
    if i == len(test):
        r.append(deepcopy(res))
        return
    getSub(test, i + 1, res,r)  # 当前位置字符不加入
    res.append(test[i])
    getSub(test, i + 1, res,r)  # 当前位置字符加入
    res.pop()
# LCS找到 匹配的子串
def getLCS(S1, S2):
    m = len(S1)
    n = len(S2)
    if m < 0 or n < 0:
        return 0
    memo = [[0] * (n + 1) for j in range(m + 1)]
    # 初始状态 第0行 第0列 都是0
    subStr = []
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if S1[i - 1] == S2[j - 1]:  # S1中的第i个字符 S2中的第j个字符
                memo[i][j] = 1 + memo[i - 1][j - 1]
                for t in subStr:
                    if t[-1][1] < i-1 and t[-1][2] < j-1:
                        t.append((S1[i - 1], i-1, j-1))
                subStr.append([(S1[i - 1], i-1, j-1)])
            else:
                memo[i][j] = max(memo[i - 1][j], memo[i][j - 1])
    return memo[m][n],subStr

import pickle
if __name__=="__main__":
    f = open('../Data/paper', 'rb')
    paper=pickle.load(f)
    # del paper['01']
    # del paper['12']
    # del paper['0803']
    # del paper['0804']
    # del paper['0823']
    fenci(paper)
    # f = open('../Data/paper2', 'rb')
    # paper2=pickle.load(f)
    # getTermRule(paper2)

    pass
