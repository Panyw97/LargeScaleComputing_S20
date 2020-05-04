import numpy as np
import scipy.stats as sts
import scipy.optimize as opt
import time
from mpi4py import MPI



def minimizer(rho, n_runs):
	# Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Evenly distribute number of simulation runs across processes
    N = int(n_runs/size)

	# Set model parameters
    mu = 3.0
    sigma = 1.0
    z_0 = mu
	# Set simulation parameters, draw all idiosyncratic random shocks, 
	# and create empty containers
    T = int(4160) # Set the number of periods for each simulation np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N))
    z_mat = np.zeros((T, N))
    z_mat[0, :] = z_0
    p = []
    for s_ind in range(N): 
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            if z_t <= 0 or t_ind == T - 1:
                p.append(np.array([t_ind]))
                break
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

# Gather all simulation arrays to buffer of expected size/dtype on rank 0
    p_all = None
    if rank == 0:
        p_all = np.empty([N * size, 1], dtype='float')
    comm.Gather(sendbuf = np.array(p), recvbuf = p_all, root=0)
    if rank == 0:
        return -np.mean(p_all)

def main():
    t0 = time.time()

    rho_init = 0.1
    results = opt.minimze(minimizer, rho_init, args=(1000))
    opt_rho = results.x
    opt_p = -results.fun

    time_elapsed = time.time() - t0
    print("The optimal rho is", opt_rho)
    print("The period is", opt_p)
    print("Computation Time:", time_elapsed)

if __name__ == '__main__':
    main()