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
1. Edit the file path of the matrices and the regularization parameter in lingo.py.
2. The command 'python lingo.py' executes the program.

### The execution time 
Execution time is within seconds.

### The expected results
The output is the parameter of the Ridge Regression model. For example, Equation 19 in our paper is one of the output.

## kernelmodel_evaluation
The SpMV method in benchmark is CSR-Adaptive in rocSPARSE without optimization strategy. It is used as a benchmark to compare the performance of our algorithm.
### The experiment workflow


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
