# encoding: utf-8
import os
import subprocess
import csv
import time
from tqdm import tqdm
#import scanpy as sc
import pandas as pd

def subprocess_popen(statement):
    p = subprocess.Popen(statement, shell=True, stdout=subprocess.PIPE)  
    while p.poll() is None: 
        if p.wait() != 0: 
            return ["Error."]
        else:
            re = p.stdout.readlines()  
            result = []
            for i in range(len(re)):  
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
        text = subprocess_popen(f"{sp_format} {namelist_mtx[i]}")
        if len(text) != 13:
            continue
        try:
            if len(text) == 13:
                writer.writerow([text[0],text[2],text[4],text[6],text[8],text[10],text[12]])
                num = num + 1
            else:
                #writer.writerow([text[0]])
                pass
        except:
            pass
    print(num)


namelist_realmtx = []
files = "/work1/wangjue/linkehao/espreso/qdx_712/20230310/001mtx/p_"
for i in range(6):
    for j in range(2048):
        filename = files + str(i) + "_d_" + str(j) + ".mtx"
        namelist_realmtx.append(filename)


EXE_getfeatures = "/public/home/wangjue/shiym/getfeatures/getfeatures"

csv_filename = "001mtx_features_pre.csv"
if (os.path.exists(csv_filename)):
    os.remove(csv_filename)

# SPMV
not_exe_mtx = []
not_acc_mtx = []

with open(csv_filename,"a") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["filenames","Rows","NNZ","Ave_R","Dis","Gini_cb","Aariance"])

    getfeatures_shell(EXE_getfeatures, namelist_realmtx, not_exe_mtx, not_acc_mtx, writer)
    time.sleep(0.01)
print("Done")

