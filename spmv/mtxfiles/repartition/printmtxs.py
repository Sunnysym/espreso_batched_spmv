# encoding: utf-8
import os
import subprocess
import csv
import time
from tqdm import tqdm
import pandas as pd
import numpy

f = open("/public/home/wangjue/shiym/spmv/mtxfiles/repartition/matrix_partition","r")
result = []
# print(f.read())
for line in f:
    line = line.strip('\n')
    result.append(line)
f.close()
result=numpy.array(result,dtype=int)
length = result[0]
mtxgroup = result[1:]
'''
data = numpy.genfromtxt(r'/public/home/wangjue/shiym/spmv/mtxfiles/repartition/001pre.csv',delimiter=',')
data = data[1:]

pretime = [0,0,0,0,0,0]
for i in range(len(mtxgroup)):
    pretime[mtxgroup[i]] = pretime[mtxgroup[i]] + data[i]

for i in range(6):
    print(pretime[i])

'''
process = [[] for i in range(6)]
files = "/work1/wangjue/linkehao/espreso/qdx_712/20230310/001mtx/p_"
for i in range(6):
    for j in range(2048):
        filename = files + str(i) + "_d_" + str(j) + ".mtx"
        id = i * 2048 + j
        process[mtxgroup[id]].append(filename)


base = "/public/home/wangjue/shiym/spmv/mtxfiles/repartition/process"
for i in range(6):
    writefile = base + str(i) + ".txt"
    fp = open(writefile, "a")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
    print(len(process[i]),file=fp)
    for j in range(len(process[i])):
        print(process[i][j], file=fp)
    fp.close()







