import numpy as np
import matplotlib.pyplot as plt  # 加载matplotilib模块
import matplotlib
import pandas as pd

from sklearn.metrics import mean_squared_error
import math
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

data = pd.read_csv("spmv1024_pipeline.csv")
data = np.array(data)
cols = data[:, 0]
prepipeline = data[:, 1]
pipeline = data[:,2]
print(len(pipeline))

mse = mean_squared_error(pipeline,prepipeline)
mae = get_mae(pipeline,prepipeline)
print(mse)
print(np.sqrt(mse))
print(mae)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize']=(5.5,3)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(range(len(pipeline)), pipeline, 'b', label="observed")
ax1.plot(range(len(prepipeline)), prepipeline, 'o', markersize=1,color='red',label='estimated')  # 展示真实数据（蓝色）以及拟合的曲线（红色）
ax1.legend(loc='lower right',frameon=True, edgecolor='k', handlelength=0.9, handleheight=0.9, fontsize=10)  # 设置图例的位置
ax1.set_xlabel("Matrix set index",fontsize=12)
ax1.set_ylabel("Time (s)",fontsize=12)
plt.subplots_adjust(left=0.14,bottom=0.16,right=0.96,top=0.94,wspace=0.2,hspace=0.2)
plt.ylim([0,0.02])
plt.show()