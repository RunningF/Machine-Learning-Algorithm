# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:58:01 2016

@author: jining

梯度下降算法
x=x-alpha*▽x
"""
import numpy as np
import pylab as pb
#梯度下降算法求解最优回归系数
#w=w-xT*(x*w-y)*alpha 矩阵乘法
#x:自变量. 行数：样本数，列数：特征数+1，加1是因为截距
#y:因变量. 行数：样本数，列数：1
#w:回归系数. 行数：特征数+1，列数：1
#alpha：步长. 1*1，标量

#梯度下降算法
#x为自变量，y为相应变量，w为回归系数，n为迭代次数，alpha为步长
#返回回归系数矩阵
def grad(x,y,n=1000,alpha=0.0001):
    n_features = x.shape[1] #包括截距
    w = np.mat(np.ones((n_features,1))) #回归系数初始化为1
    for i in range(n):
        w = w-x.transpose()*(x*w-y)*alpha
    return w

#随机生成20个数据点,并可视化
np.random.seed(1)
x = np.array(range(0,20))
y = 2*x+1+np.random.normal(0,5,len(x))
pb.plot(x,y,'ro');pb.xlim(-1,len(x))
#构建矩阵x、矩阵y
xmatrix = np.mat([[1,i] for i in x]) #增加的一列代表截距,shape:20*2
ymatrix = np.mat(y).transpose() #shape:20*1
#调用梯度下降算法，得到回归系数矩阵w，画出最佳拟合直线
w = grad(xmatrix,ymatrix)    #w[0]是截距值，其余是特征的回归系数值
pb.plot(x,y,'ro');pb.plot([0,19],[int(w[0]+w[1]*0),int(w[0]+w[1]*19)]);pb.xlim(-1,len(x))
    
'''
#梯度下降算法求解二次曲线的最小值 y=x^2-2x+3
x = np.linspace(-4,6,100)
y = x**2-2*x+3
pb.plot(x,y)

#x是初始值
def grad2(x,n=1000,alpha=0.01):
    for i in range(n):
        x=x-alpha*(2*x-2)
    return x

mx = grad2(5)
pb.plot(x,y);pb.plot(mx,mx**2-2*mx+3,'ro');pb.grid()
'''





