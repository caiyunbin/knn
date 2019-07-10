# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:13:31 2019

@author: Caiyunbin
"""


import pandas as pd
import numpy as np


data = pd.DataFrame({
        'id':np.array(range(1,8),dtype='int32'),
        '名称':pd.Series(['魂断蓝桥','泰坦尼克号','触不可及','尖峰时刻','新警察故事','少林寺','咏春']),
        '打斗镜头':np.array([1,5,10,255,304,222,663],dtype ='int32'),
        '激情镜头':np.array([298,340,276,20,30,15,8],dtype ='int32'),
        '标签':pd.Categorical(['爱情片','爱情片','爱情片','动作片','动作片','动作片','动作片'])
        })


##KNN算法  关于列表嵌套列表的写法

def distance(vector1,vector2,length):
    dist = 0
    for i in range(length):
        dist +=pow((vector1[i] - vector2[i]),2)
    return dist

def get_neibors(trainset,testset,k):
    dist = []
    length = len(testset)-1
    for i in range(len(trainset)):
        dis = distance(trainset[i],testset,length)
        dist.append((trainset[i],dis))
    dist.sort(key=lambda dist:dist[1],reverse = False)
    neibors = []
    for i in range(k):
        neibors.append(dist[i][0])
    return neibors
    
def get_result(neibors):
    vote = {}
    for i in range(len(neibors)):
        lable = neibors[i][-1]
        if lable in vote:
            vote[lable] += 1
        else:
            vote[lable] = 1
    sort_vote = sorted(vote.items(),key =lambda vote:vote[1],reverse = False)
    return sort_vote[0][0]
    

def main():
    result = []
    testset = [[23,168],[26,189],[380,36]]
    for test in testset:
        nei = get_neibors([[12,405,'a'],[13,406,'a'],[305,25,'b'],[204,12,'b'],[26,405,'a']],test,3)
        res = get_result(nei)
        print(res)
        result.append(res)
    print(result)

    
if __name__ == '__main__':
    main()
    

##使用pandas进行的knn算法
def classfy_kNN(trainset,testset,k):
    result = []
    dist=list((((trainset.iloc[:,2:4]-testset)**2).sum(1))**0.5)
    dist_l=pd.DataFrame({'dist':dist,'lable':trainset.iloc[:,-2]})
    get_neighbour = dist_l.sort_values(by ='dist')[:k]
    get_votes = get_neighbour.loc[:,'lable'].value_counts()
    result.append(get_votes.index[0])
    return result
    
if __name__ == '__main__':
    result = []
    trainset =data
    testset = [[7,206],[305,20]]
    k=4
    for test in testset:
        re = classfy_kNN(trainset,test,k)
        result.append(re)
    print(result)