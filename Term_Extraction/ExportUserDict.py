# -*- coding: utf-8 -*-
#-------------------------------------------------
#   File Name：     ExportUserDict
#   Author :        fengge
#   date：          2019/3/27
#   Description :   将词和原始的字典合并成新的词典
#-------------------------------------------------
from config import data_dir
import os
def ExporUserDict(words,infile=data_dir+"/Data/user.txt",outfile=data_dir+"/Data/user.txt",flag=True,n="n",v=100):
    '''
    将词和原始的字典合并成新的词典
    Parameter:
        - words: 添加的新词
        - infile: 原来词典的位置
        - outfile:添加的词典位置
        - flag: True的时候使用默认词频和词性，False的时候不使用
        - n: 默认词性
        - v: 默认词频
    '''
    userWord=[]
    if os.path.exists(infile):
        with open(infile,'r',encoding='utf8') as old_file:
            tmp=[t.strip() for t in old_file.readlines()]
            userWord.extend(tmp)
    suffix=' '+str(v)+' '+str(n) if flag else ""
    for w in words:
        userWord.append(w+suffix)
    newWord=set(userWord)
    new_file=open(outfile,'w',encoding='utf8')
    for w in newWord:
        new_file.write(w+"\n")
    new_file.close()
