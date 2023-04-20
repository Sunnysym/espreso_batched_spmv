# performance prediction model

## getfeatures
The code is used to get features of the sparse matrix. The features include the number of rows, the number of non-zeros, the average number of non-zeros per row, average distance between each pair of continuous non-zeros, Gini coefficient of the non-zeros number between column blocks, and variance of the number of non-zeros per row.
### The experiment workflow
1. The command 'make' generates an executable file
> **make**
2. Edit the file path of the matrices in getfeatures.py.
3. The command 'python getfeatures.py' executes the program.
### The execution time 
The execution time of the program depends on the number of matrices. The time to get features for a matrix is within one second.
### The expected results
The output is the name and features of the matrices.

## kernelmodel
The code of Ridge Regression model to build kernel execution performance model.
### The experiment workflow
1. Edit the file path of the matrices and the range of the values for regularization parameter in lingo.py.
2. The command 'python lingo.py' executes the program.
### The execution time 
Execution time is within seconds.
### The expected results
The output is the parameter of the Ridge Regression model. For example, Equation 19 in our paper is one of the output.

## kernelmodel_evaluation
The code is used to test the kernel execution performance model accuracy.
### The experiment workflow
1. Edit the file path of the result.
2. The command 'python execution.py' executes the program.
### The execution time 
Execution time is within seconds.
### The expected results
The output is the figure of the observed and estimated SpMV kernel execution time, as shown in Figure8 in our paper.

## PCIe model
It is used to get the parameters in PCIe transfer model.
### The experiment workflow
1. Edit the datatype and datasize of the input and output data in transfer.cpp.
2. The command 'make' generates an executable file
> **make**
3.  At slurm workload manager, modify run.slurm file to run. Here is an example. Set up at 1 node. The command 'sbatch run.slurm' executes the program.
```slurm
#!/bin/bash
#SBATCH -J transfer  
#SBATCH -p normal 
#SBATCH -N 1  
#SBATCH -n 1  
#SBATCH --gres=dcu:4  
#SBATCH --ntasks-per-node=1  
#SBATCH --ntasks-per-socket=1  
#SBATCH --cpus-per-task=32 
#SBATCH --mem=90G  
#SBATCH -o slurmlog/transfer.log 
#SBATCH -e slurmlog/transfer.log  
#SBATCH --exclusive 

export MASTER_PORT=25875
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo start on $(date)
echo "SLURM_JOB_ID: $SLURM_JOB_ID" 
echo "Basic Information"
APP="./transfer" 
echo "Running APP: $APP"
srun $APP
echo end on $(date)
```
### The execution time 
Execution time is within seconds.
### The expected results
The output is the data transfer time which can be used to get the parameter of PCIe transfer model.

## PCIe model evaluation
The code is used to test the PCIe data transfer model accuracy.
### The experiment workflow
1. Edit the file path in performance.py
2. The command 'python performance.py' executes the program.
### The execution time 
Execution time is within seconds.
### The expected results
The output is the figure of the kernel execution performance.

## execution evaluation 
The code is used to test the execution performance model accuracy.
### The experiment workflow
1. Edit the file path in throughput.py
2. The command 'python throughput.py' executes the program.
### The execution time 
Execution time is within seconds.
### The expected results
The output is the figure of the throughput.
