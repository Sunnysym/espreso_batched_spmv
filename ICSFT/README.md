# ICSFT
After the HTFETI two-level domain decomposition, each subdomain has been determined. The number of matrices in each cluster is storaged in dual files. We propose the Inter-cluster subdomain fine-tuning (ICSFT) algorithm to adjust the subdomains between clusters, according to the weights assigned to subdomains in the graph. The weights of each subdomain is storaged in pre.csv. The code in read_dual_and_balance is ICSFT algorithm for load balance.
### The experiment workflow
1. The command 'g++ read_dual_and_balance.cpp -o read_dual_and_balance' generates an executable file
> **g++ read_dual_and_balance.cpp -o read_dual_and_balance**
2. Run read_dual_and_balance code by the command './run.sh [matrix_num] [imbalance_ratio]' where matrix_num is the number of matrices and the imbalance_ratio is the parameter to adjust the threshold for cluster balancing weights.
> **./run.sh [matrix_num] [imbalance_ratio]**
### The execution time 
Execution time is within seconds.
### The expected results
The output is the cluster index of each subdomain after fine-tuning.


