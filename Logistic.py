# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:24:00 2016

@author: jining
"""

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

import pandas as pd

#读入数据
trainPath = r'C:\Users\jining\Desktop\机器学习资料\_源代码\machinelearninginaction\Ch05\horseColicTraining.txt'
testPath = r'C:\Users\jining\Desktop\机器学习资料\_源代码\machinelearninginaction\Ch05\horseColicTest.txt'
train = pd.read_csv(trainPath,sep='\t',header=None)
test = pd.read_csv(testPath,sep='\t',header=None)

trainFeature = train.loc[:,0:20]
trainTarget = train[21]

testFeature = test.loc[:,0:20]
testTarget = test[21]

#在训练集上用交叉验证构建默认的Logistic模型
#在训练集上进行10次10折交叉验证(shuffle)，得到平均正确率
clf = LogisticRegression()
scores = []
for i in range(10):
    score = cross_val_score(clf,trainFeature,trainTarget,cv=KFold(len(trainFeature),10,True)).mean()    
    scores.append(score)
score = sum(scores)/len(scores)

#用全部训练数据构建Logistic模型
#基于此模型预测测试数据集，得到正确率
clf.fit(trainFeature,trainTarget)
clf.score(testFeature,testTarget)

#发现测试集上正确率大于训练集的
#原因分析：
# 1.训练集上模型是分成10份，9份训练，1份获得得分；
#   而测试集上模型是用全部训练集训练，在测试集上获得得分；
#   可能数据量太少，导致问题的出现。
# 2.模型泛化能力太差，没有学到数据的真实特点
# 3.可能测试数据的分布比较特殊，不具有一般代表性
# 4.没有进行变量选择，模型学习了很多噪音

#对Logistic模型进行调参优化
#定义模型评分标准，默认进行10次10折交叉验证的平均值
def scoring(clf,x,y):
    scores = []
    for i in range(10):
        score = cross_val_score(clf,x,y,cv=KFold(len(x),10,True)).mean()    
        scores.append(score)
    return sum(scores)/len(scores)
#定义模型参数范围    
parameters = {'penalty':['l1','l2'],
              'C':[0.1,1,10],
              'max_iter':[100,500,1000]}
#建立GridSearchCV，使用自定义的scoring（默认的scoring为clf的score）
grid_search = GridSearchCV(clf,parameters,
                           scoring=scoring,
                           n_jobs = 1,verbose=1)
grid_search.fit(trainFeature,trainTarget)

#模型更新为调参之后的最优模型
clf = grid_search.best_estimator_
#用新模型获得测试集的得分
clf.fit(trainFeature,trainTarget)
clf.score(testFeature,testTarget)

#模型效果有所提升，进行特征选择
select = SelectKBest()
clf = LogisticRegression()
parameters = {'clf__penalty':['l1','l2'],
              'clf__C':[0.1,1,10],
              'clf__max_iter':[100,500,1000],
              'select__k':list(range(1,22))}
pipeline=Pipeline([('select',select),('clf',clf)]) 
grid_search = GridSearchCV(pipeline,parameters,
                           scoring=scoring,
                           n_jobs = 1,verbose=1)
grid_search.fit(trainFeature,trainTarget) 
pipeline = grid_search.best_estimator_
pipeline.score(testFeature,testTarget)   

#数据准确率从一开始的73%上升到79%                     



