import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.linear_model import Ridge  # 通过sklearn.linermodel加载岭回归方法
from sklearn.linear_model import RidgeCV
from sklearn import model_selection  # 加载交叉验证模块
import matplotlib.pyplot as plt  # 加载matplotilib模块
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures  # 通过加载用于创建多项式特征，如ab、a2、b2
import regressors.stats
from sklearn.preprocessing import StandardScaler
import sklearn
import math
import scipy.stats as st


def get_average(records):
    """
    平均值
    """
    return sum(records) / len(records)


def get_variance(records):
    """
    方差 反映一个数据集的离散程度
    """
    average = get_average(records)
    return sum([(x - average) ** 2 for x in records]) / len(records)


def get_standard_deviation(records):
    """
    标准差 == 均方差 反映一个数据集的离散程度
    """
    variance = get_variance(records)
    return math.sqrt(variance)


def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


def get_mae(records_real, records_predict):
    """
    平均绝对误差
    """
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


data = np.genfromtxt('model_128.csv',delimiter=',')
#X = data[1:, 1:7]
#57
Y = data[1:,7:57]
#56
#Y = data[1:,6:56]
#y = data[1:,58:59]
#57
dataset = data[1:,1:57]
#56
#dataset = data[1:,1:56]

mid = np.mean(Y,axis=1)
boundary = np.mean(Y,axis=1)
for i in range(len(boundary)):
    boundary[i] = boundary[i] **0.5

dellist = []
for i in range(len(dataset)):
    #57
    for j in range(6,56):
    #56
    #for j in range(5,55):
        if (dataset[i][j] < (mid[i] - boundary[i]) or  dataset[i][j] > (mid[i] + boundary[i])):
            dellist.append(i)
            break
print(dellist)
dataset_new = np.delete(dataset,dellist,0)

X = dataset_new[:,0:6]
Y = dataset_new[:,6:56]
#X = dataset_new[:,0:5]
#Y = dataset_new[:,5:55]
y =  np.mean(Y,axis=1)

low_CI_bound, high_CI_bound = st.t.interval(0.90, len(Y), loc=np.mean(Y, axis=1), scale=st.sem(Y,axis=1))

# *================4. 划分训练集和测试集=================================**
train_set_x, test_set_x, train_set_y, test_set_y = model_selection.train_test_split(X, y, test_size=0.3)
# 将所有数据划分为训练集和测试集，test_size表示测试集的比例，
# #random_state是随机数种子

from sklearn.preprocessing import StandardScaler

scaler = sklearn.preprocessing.StandardScaler()

#在训练集上使用fit_transform()
scaler.fit_transform(train_set_x)

#在测试集上使用transform()
scaler.transform(test_set_x)

alpha = []
for i in range(100):
    alpha.append(i * 0.0001 + 0.0005)
    #alpha.append(i * 0.1 + 0.5)

alphas_to_test = np.array(alpha)
rcv = RidgeCV(alphas= alphas_to_test,
    cv=None, fit_intercept=True, gcv_mode=None, normalize=False,
    scoring=None, store_cv_values=True)
rcv.fit(train_set_x,train_set_y)
smallest_idx = rcv.cv_values_.mean(axis=0).argmin()
print(alphas_to_test[smallest_idx])
print(rcv.alpha_)

import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(7, 5))
ax.set_title(r"Various values of $\alpha$")

xy = (alphas_to_test[smallest_idx], rcv.cv_values_.mean(axis=0)[smallest_idx])
xytext = (xy[0] + .01, xy[1] + .1)

ax.annotate(r'Chosen $\alpha$', xy=xy, xytext=xytext,
            arrowprops=dict(facecolor='black', shrink=0, width=0)
            )
ax.plot(alphas_to_test, rcv.cv_values_.mean(axis=0))

plt.show()
'''
# *==============5. 创建回归器，并进行训练===============================**
alphas = [0.0001,0.00001,0.0001,0.001,0.01,0.1,1,5,10,50,100,500,1000]

for a in alphas:
    clf = Ridge(alpha=a, fit_intercept=True).fit(train_set_x, train_set_y)
    score = clf.score(train_set_x, train_set_y)
    pred_y = clf.predict(train_set_x)
    mse = mean_squared_error(train_set_y,pred_y)
    print("Alpha:{0:.6f},R2:{1:.3f},MSE:{2:.2f},RMSE:{3:.2f}".format(a,score,mse,np.sqrt(mse)))
    num = clf.score(test_set_x, test_set_y)  # 利用测试集计算回归曲线的拟合优度，clf.score返回值
    print(num)
'''
a=rcv.alpha_
#a=0.01
clf = Ridge(alpha=a, fit_intercept=True)
# 接下来我们创建岭回归实例
clf.fit(train_set_x, train_set_y)  # 调用fit函数使用训练集训练回归器
print(clf.coef_)
print(clf.get_params())

#all set
print(regressors.stats.summary(clf,X,y))
num=clf.score(X, y)
pred_y = clf.predict(X)
mse = mean_squared_error(y,pred_y)
mae = get_mae(y,pred_y)
print("All set:")
print("Alpha:{0:.6f},R2:{1:.4f},MSE:{2:.4f},RMSE:{3:.4f},MAE:{4:.4f}".format(a, num, mse, np.sqrt(mse),mae))

#train set
print(regressors.stats.summary(clf,train_set_x,train_set_y))
num=clf.score(train_set_x,train_set_y)
pred_y = clf.predict(train_set_x)
mse = mean_squared_error(train_set_y,pred_y)
mae = get_mae(train_set_y,pred_y)
print("Train set:")
print("Alpha:{0:.6f},R2:{1:.4f},MSE:{2:.4f},RMSE:{3:.4f},MAE:{4:.4f}".format(a, num, mse, np.sqrt(mse),mae))

#test set
print(regressors.stats.summary(clf,test_set_x,test_set_y))
num=clf.score(test_set_x, test_set_y)  # 利用测试集计算回归曲线的拟合优度，clf.score返回值为0.7375
pred_y = clf.predict(test_set_x)
mse = mean_squared_error(test_set_y,pred_y)
mae = get_mae(test_set_y,pred_y)
print("Test set:")
print("Alpha:{0:.6f},R2:{1:.4f},MSE:{2:.4f},RMSE:{3:.4f},MAE:{4:.4f}".format(a, num, mse, np.sqrt(mse),mae))
# 拟合优度，用于评价拟合好坏，最大为1，无最小值，当对所有输入都输出同一个值时，拟合优度为0。

start = 0  # 接下来我们画一段200到300范围内的拟合曲线
end = len(X)

x_ax = range(len(X))
y_pre = clf.predict(X)  # 是调用predict函数的拟合值
time = np.arange(start, end)
#plt.plot(predicted_expect, label='estimated value')
X_part = X[start:end]
y_part = y[start:end]
y_pre_part = y_pre[start:end]
plt.plot(range(len(X_part)), y_part, 'b', label="real")
plt.plot(range(len(X_part)), y_pre_part, 'o', markersize=1,color='red',label='predict')  # 展示真实数据（蓝色）以及拟合的曲线（红色）
plt.fill_between(range(len(X_part)), low_CI_bound[start:end], high_CI_bound[start:end], alpha=0.5, label='confidence interval',color='steelblue')
plt.legend(loc='lower right')  # 设置图例的位置
plt.xlabel("Matrix Num")
plt.ylabel("GFLOPs")
plt.title('Confidence interval')
plt.show()