# Batched SpMV Library

## batch_kernel
The SpMV method in batch kernel is batched SpMV. It is used to test the kernel execution performance.
### The experiment workflow
1. Edit the parameter in test_csrspmv_batch.cpp. The variable StreamNum is the number of the stream used for SpMV. The variable batchMatNum is the size of each batch. The IterMax is the run times.
```C
int DeviceNum = 1;
int StreamNum = 4;
vector<GPUThreadArgument<double>> args;
...
int main(int argc, char* argv[])
{
    DeviceNum = 1;
    args.resize(DeviceNum);
    gpuBoot();

    int IterMax = 102;
    int MatNumMax = 1024;

    int nnznum = 0;
    int rownum = 0;
    int colnum = 0;

    int batchMatNum = 256;

    int Diffmtxnum = 1;
    int ifmtxs = 1;
 
    vector<string> mtxnames;
...
}
```
2. Set rocSPARSE path in the Makefile
3. The command 'make' generates an executable file
> **make**
4. At slurm workload manager, modify run.slurm file to run. Here is an example. Set up at 1 node. The command 'sbatch run.slurm' executes the program.
```slurm
#!/bin/bash
#SBATCH -J kernel  
#SBATCH -p normal 
#SBATCH -N 1  
#SBATCH -n 1  
#SBATCH --gres=dcu:4  
#SBATCH --ntasks-per-node=1  
#SBATCH --ntasks-per-socket=1  
#SBATCH --cpus-per-task=32 
#SBATCH --mem=90G  
#SBATCH -o slurmlog/kernel.log 
#SBATCH -e slurmlog/kernel.log  
#SBATCH --exclusive 


export MASTER_PORT=25875
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo start on $(date)
echo "SLURM_JOB_ID: $SLURM_JOB_ID" 
echo "Basic Information"
APP="./1024batch_kernel_50 process0.txt"  
echo "Running APP: $APP"
srun $APP
echo end on $(date)
```

### The execution time 
The execution time of the program depends on the number of matrices and the number of executions to perform the sparse matrix-vector multiplication. 4096 matrices each perform matrix-vector multiplication with the parameter of 1024 batch size and 4 stream num within 0.013s.

### The expected results
Line 1 output the kernel execution time

Line 2 output the floating point performance of the kernel execution 

Line 3 output the kernel execution time and vector transfer time

We mainly use the floating point performance of the kernel execution. It is the result shown in Figure 4: Performance of SpMV schemes comparing batches and streams.

## batch_pipeline
The SpMV method in batch pipeline is batched SpMV. It is used to test the throughput. It is the main component of the Batched SpMV Library.
### The experiment workflow
1. Edit the parameter in test_csrspmv_pipeline.cpp. The variable StreamNum is the number of the stream used for SpMV. The variable batchMatNum is the size of each batch. The IterMax is the run times.
```C
int DeviceNum = 1;
int StreamNum = 4;
vector<GPUThreadArgument<double>> args;
...
int main(int argc, char* argv[])
{
    DeviceNum = 1;
    args.resize(DeviceNum);
    gpuBoot();

    int IterMax = 102;
    int MatNumMax = 1024;

    int nnznum = 0;
    int rownum = 0;
    int colnum = 0;

    int batchMatNum = 256;

    int Diffmtxnum = 1;
    int ifmtxs = 1;
 
    vector<string> mtxnames;
...
}
```
2. Set rocSPARSE path in the Makefile
3. The command 'make' generates an executable file
> **make**
4. At slurm workload manager, modify run.slurm file to run. Here is an example. Set up at 1 node. The command 'sbatch run.slurm' executes the program.
```slurm
#!/bin/bash
#SBATCH -J pipeline  
#SBATCH -p normal 
#SBATCH -N 1  
#SBATCH -n 1  
#SBATCH --gres=dcu:4  
#SBATCH --ntasks-per-node=1  
#SBATCH --ntasks-per-socket=1  
#SBATCH --cpus-per-task=32 
#SBATCH --mem=90G  
#SBATCH -o slurmlog/pipeline.log 
#SBATCH -e slurmlog/pipeline.log  
#SBATCH --exclusive 


export MASTER_PORT=25875
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo start on $(date)
echo "SLURM_JOB_ID: $SLURM_JOB_ID" 
echo "Basic Information"
APP="./pipeline_1batch_5stream process0.txt"  
echo "Running APP: $APP"
srun $APP
echo end on $(date)
```

### The execution time 
The execution time of the program depends on the number of matrices and the number of executions to perform the sparse matrix-vector multiplication. 4096 matrices each perform matrix-vector multiplication with the parameter of 1024 batch size and 4 stream num within 0.01s.

### The expected results
Line 1 output the kernel execution time and vector transfer time 

Line 2 output the throughput 

We mainly use the throughput. It is the result shown in Figure 5: Throughput of SpMV schemes comparing batches and streams.

## benchmark
The SpMV method in benchmark is CSR-Adaptive in rocSPARSE without optimization strategy. It is used as a benchmark to compare the performance of our algorithm.
### The experiment workflow
1. Set rocSPARSE path in the Makefile
2. The command 'make' generates an executable file
> **make**
3. At slurm workload manager, modify run.slurm file to run. Here is an example. Set up at 1 node. The command 'sbatch run.slurm' executes the program.
```slurm
#!/bin/bash
#SBATCH -J benchmark  
#SBATCH -p normal 
#SBATCH -N 1  
#SBATCH -n 1  
#SBATCH --gres=dcu:4  
#SBATCH --ntasks-per-node=1  
#SBATCH --ntasks-per-socket=1  
#SBATCH --cpus-per-task=32 
#SBATCH --mem=90G  
#SBATCH -o slurmlog/benchmark.log 
#SBATCH -e slurmlog/benchmark.log  
#SBATCH --exclusive 


export MASTER_PORT=25875
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo start on $(date)
echo "SLURM_JOB_ID: $SLURM_JOB_ID" 
echo "Basic Information"
APP="./benchmark_50 process0.txt"  
echo "Running APP: $APP"
srun $APP
echo end on $(date)
```

### The execution time 
The execution time of the program depends on the number of matrices and the number of executions to perform the sparse matrix-vector multiplication. 4096 matrices each perform matrix-vector multiplication within 2s.

### The expected results
Line 1 output the kernel execution time

Line 2 output the floating point performance of the kernel execution 

Line 3 output the kernel execution time and vector transfer time

We mainly use the floating point performance of the kernel execution. The base time in Figure 11: Weak scalability results and Figure 12: Strong scalability results can be get through executing the benchmark code.

## mtxfiles
It is used to print matrix files that need to perform sparse matrix-vector multiplication.
### The experiment workflow
1. Edit the file path in printmtxs.py
2. The command 'python printmtxs.py' executes the program.
### The execution time 
Execution time is within seconds.
### The expected results
The output is the matrix files that need to perform sparse matrix-vector multiplication.

## kernel execution performance
It is used to visualize kernel execution performance test results.
### The experiment workflow
1. Edit the file path in performance.py
2. The command 'python performance.py' executes the program.
### The execution time 
Execution time is within seconds.
### The expected results
The output is the figure of the kernel execution performance.

## throughput 
It is used to visualize throughput test results.
### The experiment workflow
1. Edit the file path in throughput.py
2. The command 'python throughput.py' executes the program.
### The execution time 
Execution time is within seconds.
### The expected results
The output is the figure of the throughput.
