#!/bin/bash

#SBATCH --job-name=minimizer_mpi_2
#SBATCH --output=minimizer_mpi_2.out
#SBATCH --ntasks=20
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

# Load the default mpi4py/Anaconda module.
module load mpi4py/3.0.1a0_py3

# Run the python program with mpirun. The -n flag is not required;
# mpirun will automatically figure out the best configuration from the
# Slurm environment variables.
mpirun -n 20 python ./minimizer_mpi_2.py > ./minimizer_mpi_2.out
