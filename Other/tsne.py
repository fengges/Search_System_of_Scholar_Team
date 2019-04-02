# -*- coding: utf-8 -*-
#-------------------------------------------------
#   File Name：     tsne
#   Author :        fengge
#   date：          2019/3/31
#   Description :   
#-------------------------------------------------

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pickle,random
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ['FangSong']
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())

sorted_names = [name for hsv, name in by_hsv]
# 最多显示20个，可以自己添加序号,在color_num图片查看
need=[0,5,16,21,23,28,35,41,56,70,72,100,103,113,114,124,130,133,140,142]
sorted_names=[name  for index,name in  enumerate(sorted_names) if index in need]
# random.shuffle(sorted_names)
def getPic(perfex,p):
    file=open("../Data/"+perfex+"_"+str(p)+".txt",'r',encoding='utf8')
    lines=file.readlines()
    paper_list=[]
    code=set()
    for line in lines:
        dict=eval(line)
        code.add(dict['DISCIPLINE_NAME'])
        paper_list.append([dict['id'],dict['DISCIPLINE_NAME'],dict['vec']])
    file.close()
    del lines
    code_color={}
    start=0
    for i,c in enumerate(code):
        # if start>10:
        #     break
        # start+=1
        code_color[c]=colors[sorted_names[i]]
    plt.figure(figsize=(16, 9), dpi=120)

    for i in code_color:
        label = i
        size = 20.0
        linewidths = 0.5
        color = code_color[label]
        X1=[p[2][0] for p in paper_list if p[1]==label]
        X2=[p[2][1] for p in paper_list if p[1]==label]
        X1=[x for t,x in enumerate(X1) if t % 3 == 0]
        X2=[x for t,x in enumerate(X2) if t % 3 == 0]
        plt.scatter(
            X1,
            X2,
            s=size, c=color,
            linewidths=linewidths,
            label=label
        )
    leg = plt.legend()
    leg.get_frame().set_alpha(0.3)
    plt.savefig(perfex+"_"+str(p)+".png")
    # plt.show()
perplexitys=[5,10,20,30,50,100,200]
for perplexity in perplexitys:
    getPic(perplexity)