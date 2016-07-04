# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 11:44:12 2016

@author: jining
"""

import pandas as pd
import pylab as pb
from sklearn.decomposition import PCA

#读入数据
#数据有1567个样本，590个特征
path=r'\secom.data'
data=pd.read_csv(path,header=None,sep=' ')

#利用pca算法进行降维
#通过pca观察各主成分贡献的分差百分比
pca=PCA()
pca.fit(data)
#画出前20个最重要的主成分所贡献的分差百分比
pb.plot(pca.explained_variance_ratio_[0:20],'ro-')
#画出前20个累计主成分分差百分比
acus=[]
s=0
for i in pca.explained_variance_ratio_:
    s=s+i
    acus.append(s)
pb.plot(acus[0:20],'ro-');pb.grid()   

#通过acus可以看出前6个主成分贡献了96.8%的分差
#因此可以重新调用pca算法将数据6降为6维度
pca=PCA(n_components=6)
reduced_data = pca.fit_transform(data)



    
    
