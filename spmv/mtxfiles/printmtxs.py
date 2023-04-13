# encoding: utf-8
import os
import subprocess
import csv
import time
from tqdm import tqdm
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
        if path[-3:] == "mtx" and path[58] == "5":
            if os.path.isfile(path):
                _files.append(path)
    return _files

def getmtx(sp_format, namelist_mtx, not_exe_mtx, not_acc_mtx):
    num = 0
    for i in tqdm(range(len(namelist_mtx))):
    #for i in tqdm(range(2)):    
        subprocess_popen(f"{sp_format} {namelist_mtx[i]}")
        num = num + 1 
        print(num)


#namelist_realmtx = list_all_files("/hdd1/dataset_spmv/real_mtx/")
#namelist_realmtx = list_all_files("/hdd1/nas_157/OGB/ogbn_partitions/")
namelist_realmtx = list_all_files("/work1/wangjue/linkehao/espreso/qdx_712/20230310/001mtx/")

#namelist_realmtx[:8]
#print(len(namelist_realmtx))

#EXE_csr = "/public/home/wangjue/shiym/getmtx/getmtx"


# SPMV
not_exe_mtx = []
not_acc_mtx = []

#print(namelist_realmtx)
#getmtx(EXE_csr, namelist_realmtx, not_exe_mtx, not_acc_mtx)

fp = open("/public/home/wangjue/shiym/spmv/mtxfiles/process0.txt", "a")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
for i in range(len(namelist_realmtx)):
    print(namelist_realmtx[i], file=fp)
fp.close()


'''
with open(csv_filename,"a") as csvfile:
    writer = csv.writer(csvfile)

    #先写入columns_name
    #writer.writerow(["mtxname","row","col","nnz", "data1","data2","data3","data4","data5","data6","data7","data8","data9","data10","data11","data12","data13","data14","data15","data16","data17","data18","data19","data20","data21","data22","data23","data24","data25","data26","data27","data28","data29","data30","data31","data32","scoo_sparsebar","clusters","clusters_blocknum","ratio","col_clusters","col_clusters_blocknum","col_ratio" ])
    writer.writerow(["mtxname","row","col","nnz","cusparse_csr time","small_scoo time","scoo time_tuning","COO","CSR","SCOO","block_length","kernel1","dim1","kernel2","dim2","kernel3","dim3"])
    #writer.writerow(["mtxname","row","col","nnz","kernel1 time","kernel2 time","kernel3 time"])
    #写入多行用writerows
#     n_exe_mtx_bmcoo, n_acc_mtx_bmcoo = spmv_shell_bmcoo(EXE_spmv_coo, namelist_realmtx, not_exe_mtx, not_acc_mtx, writer)
    spmv_shell_scoo(EXE_spmv, namelist_realmtx, not_exe_mtx, not_acc_mtx, writer)
    time.sleep(0.01)
print("Done")


# print
#for i in range(len(namelist_realmtx)):
for i in range(2):
    print(f"{EXE_spmv} {namelist_realmtx[i]}")
    #print("{} {}".format(EXE_spmm_256,namelist_realmtx[200]))
    #text = subprocess_popen("{} {}".format(EXE_spmm_256,namelist_realmtx[200]))


    text = subprocess_popen(f"{EXE_spmv} {namelist_realmtx[i]}")
    #a = "/home/shiym/spmm/mtx_data/pdb1HYS.mtx"
    #text = subprocess_popen(f"{EXE_spmm_256} {a}")

    print(len(text))
    for i in range(len(text)):
        print('The number is %d'%i)
        print(text[i])
'''
