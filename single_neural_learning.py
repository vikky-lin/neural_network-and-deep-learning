import numpy as np
from random import random
import matplotlib.pyplot as plt 


"""
* @author:vikky
* desc:对单个sigmoid神经元应用梯度下降法进行权重学习，为简化问题，省去偏置项
"""

def sigmoid(x):
    """
    * sigmoid函数
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    * sigmoid函数导数
    """
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    """
    * 双曲正切函数
    """
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_prime(x):
    """
    * tanh函数导数
    """
    return 1-tanh(x)**2

def ReLU(x):
    """
    * 修正线性神经元函数
    """
    return max(0,x)

def ReLU_prime(x):
    """
    * ReLU导数
    """
    return 1 if x>0 else 0

def softplus(x):
    """
    * sigmoid的原函数
    """
    return np.log(1+np.exp(x))

def softplus_prime(x):
    """
    * softplus导数
    """
    return sigmoid(x)

def calc_w_input(input,weight):
    """
    * 计算神经元函数的带权输入，即w*x
    """
    return input.dot(weight.T)

def loss_func(target,output,method='ms'):
    """
    * 计算损失值
    """
    if method == 'ms':  # 均方损失
        return ((target-output)**2)/2   
    if method == 'en':   # 交叉熵损失
        return -(target*np.log(output)+(1-target)*np.log(1-output))
        


def update(input,weight,w_input,error,learning_rate,method='ms',func='sigmoid'):
    """
    * 使用梯度下降法进行权重更新,待修改
    """
    if func == 'sigmoid':
        if method == 'ms':
            return  weight+learning_rate*error*sigmoid_prime(w_input)*input
        if method == 'en':
            return weight+learning_rate*error*input
    elif func == 'tanh':
        if method == 'ms':
            return  weight+learning_rate*error*sigmoid_prime(w_input)*input
        if method == 'en':
            return weight+learning_rate*error*((1+tanh(w_input))/tanh(w_input))*input        


def main(input,weight,target,learning_rate,epoch,method='ms',func='sigmoid'):
    w_input = calc_w_input(input,weight)
    if func == 'sigmoid':
        output = sigmoid(w_input)   # 计算神经元输出值
    if func == 'tanh':
        output = tanh(w_input)
    error = target-output   # 计算当前样本输出值与目标值的误差
    loss = loss_func(target,output,method)  # 根据损失函数计算当前样本输出值的损失值
    X = []  # X轴
    Y = []  # Y轴
    for e in range(epoch):
        X.append(e)
        Y.append(output)
        weight = update(input,weight,w_input,error,learning_rate,method)    # 权重更新
        w_input = calc_w_input(input,weight)
        if func == 'sigmoid':
            output = sigmoid(w_input)   # 计算神经元输出值
        if func == 'tanh':
            output = tanh(w_input)
        error = target-output
        loss = loss_func(target,output,method)
    return X,Y

if __name__ == '__main__':
    weight = np.array([1,1,1])
    input = np.array([1,1,1])
    learning_rate = 0.2
    target = 0
    epoch = 100
    X,Y = main(input,weight,target,learning_rate,epoch,method='en',func='tanh')
    # X,Y1 = main(input,weight,target,learning_rate,epoch,method='en',func='sigmoid')
    plt.plot(X,Y,label='tanh')
    # plt.plot(X,Y1,label='sigmoid')
    plt.legend()
    plt.show()

