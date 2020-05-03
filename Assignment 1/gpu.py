import numpy as np
import pyopencl as cl
import scipy.stats as sts
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel
import time


def sim_life_parallel(n_runs):
	# Set up context and command queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mem_pool = cltools.MemoryPool(cltools.ImmediateAllocator(queue))

    # Start time:
    t0 = time.time()

	# Set model parameters
    rho = 0.5
    mu = 3.0
    sigma = 1.0
    z_0 = mu
	# Set simulation parameters, draw all idiosyncratic random shocks, 
	# and create empty containers
    T = int(4160) # Set the number of periods for each simulation np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, n_runs))
    z_mat = np.zeros((T, n_runs))
    z_mat[0, :] = z_0
    for s_ind in range(n_runs): 
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

    time_elapsed = time.time() - t0
    print(time_elapsed)
    return

def main():
    sim_life_parallel(n_runs = 1000)

if __name__ == '__main__':
    main()