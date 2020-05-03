import numpy as np
import scipy.stats as sts
import time
import matplotlib.pyplot as plt
from mpi4py import MPI


def sim_life_parallel(n_runs):
	# Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start time:
    t0 = time.time()

    # Evenly distribute number of simulation runs across processes
    N = int(n_runs/size)

	# Set model parameters
    rho = 0.5
    mu = 3.0
    sigma = 1.0
    z_0 = mu
	# Set simulation parameters, draw all idiosyncratic random shocks, 
	# and create empty containers
    T = int(4160) # Set the number of periods for each simulation np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N))
    z_mat = np.zeros((T, N))
    z_mat[0, :] = z_0
    for s_ind in range(N): 
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

    time_elapsed = time.time() - t0
    print(size, ':', time_elapsed)
    return

def main():
    sim_life_parallel(n_runs = 1000)

if __name__ == '__main__':
    main()