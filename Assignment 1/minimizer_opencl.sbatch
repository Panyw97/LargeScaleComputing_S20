#!/bin/bash
#SBATCH --job-name=minimizer_opencl   # job name
#SBATCH --output=minimizer_opencl.out # output log file
#SBATCH --error=minimizer_opencl.err  # error file
#SBATCH --time=00:05:00  # 5 minutes of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

module load cuda
module load mpi4py/3.0.1a0_py3

python ./minimizer_opencl.py
