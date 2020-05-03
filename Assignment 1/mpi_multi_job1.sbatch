#!/bin/bash

#SBATCH --job-name=mpi_multi_job
#SBATCH --ntasks=3
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

# Load the default mpi4py/Anaconda module.
module load mpi4py/3.0.1a0_py3

# Run the python program with mpirun, using & to run jobs at the same time
for (( i = 1; i < 4; i++ )); 
do
	mpirun -n $i python ./clock.py >> ./mpi.out
done

# Wait until all simultaneous mpiruns are done
wait
