# encoding: utf-8
import os
import subprocess
import csv
import time
from tqdm import tqdm
#import scanpy as sc
import pandas as pd

def subprocess_popen(statement):
    p = subprocess.Popen(statement, shell=True, stdout=subprocess.PIPE)  # 执行shell语句并定义输出格式
    while p.poll() is None:  # 判断进程是否结束（Popen.poll()用于检查子进程（命令）是否已经执行结束，没结束返回None，结束后返回状态码）
        if p.wait() != 0:  # 判断是否执行成功（Popen.wait()等待子进程结束，并返回状态码；如果设置并且在timeout指定的秒数之后进程还没有结束，将会抛出一个TimeoutExpired异常。）
            return ["Error."]
        else:
            re = p.stdout.readlines()  # 获取原始执行结果
            result = []
            for i in range(len(re)):  # 由于原始结果需要转换编码，所以循环转为utf8编码并且去除\n换行
                res = re[i].decode('utf-8').strip('\r\n')
                result.append(res)
            return result

def list_all_files(rootdir):
    import os
    _files = []
    
    listdir = os.listdir(rootdir)
    for i in range(0, len(listdir)):
        
        path = os.path.join(rootdir, listdir[i])
        
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if path[-3:] == "mtx":
            if os.path.isfile(path):
                _files.append(path)
    return _files

def getfeatures_shell(sp_format, namelist_mtx, not_exe_mtx, not_acc_mtx, writer):
    num = 0
    for i in tqdm(range(len(namelist_mtx))):
    #for i in tqdm(range(2)):
        #if i == 3207 or i == 103 or i ==10 or i == 1312 or i == 3654:
            #continue
        text = subprocess_popen(f"{sp_format} {namelist_mtx[i]}")
        if len(text) != 17:
            continue
        #adata = sc.read(namelist_mtx[i])
        #data = adata.X
        #rows, cols = data.get_shape()
        #if rows > 100000000 or cols > 100000000:
        #if rows < 100 or cols < 100 or rows > 10000000 or cols > 10000000:
            #continue
        #del adata
        #del data
        try:
            #all is 95
            if len(text) == 17:
                '''
                writer.writerow([text[0],text[3],text[5],text[7],text[11],text[13],text[15],text[17],text[19],text[21],text[23],text[25],\
                    text[28],text[30],text[32],text[34],text[36],text[38],text[40],text[42],\
                    text[46],text[48],text[50],text[52],text[54],text[56],text[58],text[60],\
                    text[63],text[65],text[67],text[69],text[71],text[73],text[75],text[77],\
                    text[80],text[82],text[84],text[86],text[88],text[90],text[92],text[94]])
                '''
                writer.writerow([text[0],text[1],text[2],text[6],text[8],text[10],text[12],text[14],text[16]])
                num = num + 1
            else:
                #writer.writerow([text[0]])
                pass
        except:
            pass
    print(num)


#namelist_realmtx = list_all_files("/hdd1/dataset_spmv/real_mtx/")
#namelist_realmtx = list_all_files("/work1/wangjue/linkehao/espreso/qdx_712/20230310/001mtx/")
namelist_realmtx = []
files = "/public/home/wangjue/shiym/spmv/mtxfiles/process"
for i in range(6):
    filename = files + str(i) + ".txt"
    namelist_realmtx.append(filename)


#namelist_realmtx[:8]
#print(len(namelist_realmtx))

#selectmtx = pd.read_csv("newmtx.csv")
#namelist_realmtx = selectmtx["mtxname"]

EXE_benchmark = "/public/home/wangjue/shiym/spmv/benchmark/test_2048mtx"

csv_filename = "001mtx_benchmark.csv"
if (os.path.exists(csv_filename)):
    os.remove(csv_filename)

# SPMV
not_exe_mtx = []
not_acc_mtx = []

with open(csv_filename,"a") as csvfile:
    writer = csv.writer(csvfile)
    #先写入columns_name

    writer.writerow(["processname","Rownum","Colnum","NNZnum","kerneltime","kernelperformance","alltime"])

    getfeatures_shell(EXE_benchmark, namelist_realmtx, not_exe_mtx, not_acc_mtx, writer)
    time.sleep(0.01)
print("Done")

'''
#for i in range(len(namelist_realmtx)):
for i in range(2):
    print(f"{EXE_benchmark} {namelist_realmtx[i]}")

    text = subprocess_popen(f"{EXE_benchmark} {namelist_realmtx[i]}")

    print(len(text))
    for i in range(len(text)):
        print('The number is %d'%i)
        print(text[i])

'''
