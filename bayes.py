# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:05:07 2016

@author: jining
"""

import os
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score

#读入垃圾邮件和正常邮件
spam=[]
ham=[]
spamPath = r'C:\Users\jining\Desktop\机器学习资料\_源代码\machinelearninginaction\Ch04\email\spam'+'\\'
hamPath = r'C:\Users\jining\Desktop\机器学习资料\_源代码\machinelearninginaction\Ch04\email\ham'+'\\'
spamNames = os.listdir(spamPath)
hamNames = os.listdir(hamPath)
spamTarget = [1]*len(spamNames)
hamTarget = [0]*len(hamNames)
for i in spamNames:
    file = open(spamPath+i)
    data = file.read()
    file.close()
    spam.append(data)
for i in hamNames:
    file = open(hamPath+i)
    data = file.read()
    file.close()
    ham.append(data)

#自定义分词器
def tokenizer(x):
    pattern = re.compile('\W+')
    wordVec = pattern.split(x)
    wordVec = [i.lower() for i in wordVec if len(i)>1 and i.isalpha()]
    return wordVec

#将垃圾邮件内容和正常邮件内容合并，将垃圾邮件标记和正常邮件标记合并      
doc = spam + ham
target = spamTarget + hamTarget

#将所有邮件内容转换为CountVectorizer和TfidfVectorizer
#传入的doc为list，每一项是一篇邮件内容      
cv = CountVectorizer(tokenizer=tokenizer) 
vec = cv.fit_transform(doc).toarray()
tfidf = TfidfVectorizer(tokenizer=tokenizer)
tfidfVec = tfidf.fit_transform(doc).toarray()

#分别用CountVectorizer和TfidfVectorizer使用贝叶斯分类器分类,进行10折交叉验证
#MultinomialNB适合进行文本分类
clf = MultinomialNB()
cross_val_score(clf,vec,target,cv=10)
cross_val_score(clf,tfidfVec,target,cv=10)
    
    
    
    
    
    