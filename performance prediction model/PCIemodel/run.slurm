#!/bin/bash
#SBATCH -J test   #作业名称
#SBATCH -p normal  #作业提交队列
#SBATCH -N 1  #作业申请节点数
#SBATCH -n 1  #总计运行的任务数量（进程数）
#SBATCH --gres=dcu:4  #每个节点使用通用资源名称及数量,指定每个节点使用4块DCU卡
#SBATCH --ntasks-per-node=1  #每个节点运行进程数
#SBATCH --ntasks-per-socket=1  #指定每个Socket的进程数，对应于NUMA node
#SBATCH --cpus-per-task=32  #指定每个进程的CPU数，对应多线程场景
#SBATCH --mem=90G  #占用节点全部内存
#SBATCH -o slurmlog/test16  #指定作业标准输出文件的名称
#SBATCH -e slurmlog/test16  #指定作业标准错误输出文件的名称
#SBATCH --exclusive #指定作业独占计算节点

## 需要跑的程序，通常是并行的
export MASTER_PORT=25875
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo start on $(date)
echo "SLURM_JOB_ID: $SLURM_JOB_ID" 
echo "Basic Information"
APP="./transfer"  
echo "Running APP: $APP"
srun $APP
echo end on $(date)

