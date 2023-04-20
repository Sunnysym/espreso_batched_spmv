# espreso_spmv

## Install

Compiler: OpenMPI or MPICH

### Download

Download Math Lib: 

- [Intel MKL](https://software.intel.com/en-us/intel-mkl) 

MKL install config: slient.cfg

Download Graph partitioners:

- [ParMetis](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview)
- [Metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)

Metis config: metis.h (For 64 bits)

Download ROCM（For device usage)：

- [ROCM](https://github.com/RadeonOpenCompute/ROCm)
- [rocSPARSE](https://github.com/ROCmSoftwarePlatform/rocSPARSE)

### Rely Install

Install MKL:

```shell
$  tar zxvf l_mkl_2019.5.281.tgz 
$  l_mkl_2019.5.281/install.sh -s silent.cfg            # install mkl
$  intel/bin/compilervars.sh intel64                    # mkl environment for compile espreso
```

Install Metis:

```shell
$ tar zxvf metis-5.1.0.tar.gz
$ cp metis.h metis-5.1.0/include/
$ make config prefix=~/install-path/metis
$ make 
$ make install
```

Install ParMetis:

```shell
$ tar zxvf parmetis-4.0.3.tar.gz
$ cp metis.h parmetis-4.0.3/metis/include/
$ make config prefix=~/install-path/parmetis
$ make 
$ make install
```

If use device, ROCm and rocBlas install:

[ROCm Installation documentation](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

[rocSPARSE documentation](https://rocsparse.readthedocs.io/en/rocm-5.2.0/usermanual.html?highlight=iNSTALL#building-and-installing)


### Build

```bash
$ export CPATH="/home/root/install/metis/include:/home/root/install/parmetis/include:$CPATH"
$ export LD_LIBRARY_PATH="/home/root/install/metis/lib:/home/root/install/parmetis/lib:$LD_LIBRARY_PATH"
$ export CPATH="/opt/rocm/include:/opt/rocm/rocsparse/include:$CPATH"
$ export LD_LIBRARY_PATH="/opt/rocm/lib:opt/rocm/rocsparse/lib:$LD_LIBRARY_PATH"
$ cd espreso_spmv
$ chmod 755 waf
$ ./waf configure --intwidth=64 --metis=~/install-path/metis --parmetis=~/install-path/parmetis
$ ./waf -j$(nproc)
$ export PATH="/home/root/espreso_spmv/build:$PATH"
$ export LD_LIBRARY_PATH="/home/root/espreso_spmv/build:$LD_LIBRARY_PATH"
```

If use device, with ROCm and rocSPARSE to build:

```bash
$ ./waf configure --intwidth=64 --metis=/home/root/install/metis --parmetis=/home/root/install/parmetis --cxxflags='-DUSE_DEVICE -D__HIP_PLATFORM_HCC__' --linkflags="-L/opt/rocm/lib -L/opt/rocm/rocblas/lib -lrocsparse "   
$ ./waf $(nproc)  # last step link allow error
$ cd build
$ /usr/bin/mpic++ -fopenmp src/app/espreso.cpp.37.o -o/home/root/espreso_m/build/espreso -Wl,-Bstatic,--start-group -Wl,--end-group -Wl,-Bdynamic -Wl,--no-as-needed -L. -L/opt/rocm/lib -L/opt/rocm/rocblas/lib -L/home/root/install/metis/lib -L/home/root/install/parmetis/lib -lnbesinfo -lnbconfig -lnbbasis -lnbwmpi -lnbmesh -lnbinput -lnboutput -lnbwpthread -lnbwcatalyst -lnbwhdf5 -lnbwgmsh -lnbwnglib -lnbwmetis -lnbwparmetis -lnbwscotch -lnbwptscotch -lnbwkahip -lnbphysics -lnbdevel -lnbmath -lnbautoopt -lnbwmkl -lnbwcuda -lnbwhypre -lnbwmklpdss -lnbwpardiso -lnbwsuperlu -lnbwwsmp -lnbwcsparse -lnbwbem -lnbwnvtx -lnbfeti -lparmetis -lmetis -lmkl_intel_ilp64 -lmkl_core -lmkl_gnu_thread -lmkl_blacs_intelmpi_ilp64 -lrocsparse
```

## Case Run

```slurm
#!/bin/bash
#SBATCH -J %j
#SBATCH -p normal
#SBATCH -t 03:00:00             # work time
#SBATCH --mem=100G              # node mem limit
#SBATCH -n 8                    # mpi processes num
#SBATCH --ntasks-per-node=1     # mpi processes num at each node
#SBATCH --ntasks-per-socket=1   
#SBATCH --cpus-per-task=32      # cpu num at each mpi process
#SBATCH -o slurmlog/espreso.%j.log
#SBATCH -e slurmlog/espreso.%j.log
##SBATCH --gres=dcu:4           # special device node with device count

cpu_per_task=8
process_sum=${SLURM_NTASKS}
if [ -n "${SLURM_CPUS_PER_TASK}" ]; then
        cpu_per_task=$SLURM_CPUS_PER_TASK
fi
export OMP_NUM_THREADS=${cpu_per_task}

source ~/espreso_m/build/env/threading.default ${cpu_per_task}                                              
mpirun --bind-to none -n ${process_sum} espreso -c espreso.ecf 0 0 0
```

Strategy args run:  

```shell
$ mpirun -n 8 espreso -c espreso.ecf 0 0 0   # for default strategy run
$ mpirun -n 8 espreso -c espreso.ecf 1 0 0   # for coarse load-balance run
$ mpirun -n 8 espreso -c espreso.ecf 1 1 0   # for fine load-balance run
$ mpirun -n 8 espreso -c espreso.ecf 1 1 1   # for device run
```

espreso configure file information：

```ecf
INPUT {
  FORMAT   VTK_LEGACY;
  PATH         [ARG0];
  LOADER   POSIX;
  # NPATH  PRESSION;

  KEEP_MATERIAL_SETS     FALSE;
}
...
```

## Reference

[espreso](https://github.com/It4innovations/espreso)
