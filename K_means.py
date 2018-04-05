import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

def dist(x,y):
    # dist: 求样本间的余弦相似度
    return sum(x*y)/(sum(x**2)*sum(y**2))**0.5

def K_means(data=iris.data,k=3,ping=0,maxiter=100):
    # k: 类别数
    # maxiter: 最大迭代次数
    n, m = data.shape    # n:样本个数,m:属性个数
    centers = data[:k,:] #初始类中心
    while ping < maxiter:
        dis = np.zeros([n,k+1])  #距离矩阵
        for i in range(n):       #求各样本至各类中心的距离
            for j in range(k):
                dis[i,j] = dist(data[i,:],centers[j,:])
            dis[i,k] = dis[i,:k].argmax()

        centers_new = np.zeros([k,m])
        for i in range(k):
            # 求新类中心:各类样本均值作为新类中心
            index = dis[:,k]==i
            centers_new[i,:] = np.mean(data[index,:],axis=0)
        if np.all(centers==centers_new):
            # 判定类中心是否发生变化
            break
        centers = centers_new #更新类中心
        ping += 1
    return dis
if __name__ == '__main__':
    res = K_means()
    print(res)
