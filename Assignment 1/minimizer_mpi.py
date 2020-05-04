import numpy as np
import scipy.stats as sts
import scipy.optimize as opt
import time
from mpi4py import MPI



def minimizer(n_runs):
    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    t0 = time.time()
    # Evenly distribute number of simulation runs across processes
    N = int(n_runs/size)

    # Set model parameters
    mu = 3.0
    sigma = 1.0
    z_0 = mu
    # Set simulation parameters, draw all idiosyncratic random shocks, 
    # and create empty containers
    T = int(4160) # Set the number of periods for each simulation np.random.seed(25)
    def parallel_function_caller(rho):
        eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(N, T))
        z_mat = np.zeros((N, T))
        z_mat[:, 0] = z_0
        p = []
        for s_ind in range(N): 
            z_tm1 = z_0
            for t_ind in range(T):
                e_t = eps_mat[s_ind, t_ind]
                z_t = rho * z_tm1 + (1 - rho) * mu + e_t
                if z_t <= 0 or t_ind == T - 1:
                    p.append(np.array([t_ind]))
                    break
                z_mat[s_ind, t_ind] = z_t
            z_tm1 = z_t
        p_ary = np.array(p)
# Gather all simulation arrays to buffer of expected size/dtype on rank 0
        p_all = None
        if rank == 0:
            p_all = np.empty([N * size, 1], dtype='float')
        comm.Gather(sendbuf = p_ary, recvbuf = p_all, root=0)
        if rank == 0:
            return -np.mean(p_all)
    if rank == 0:
        rho_init = 0.1
        results = opt.minimize(parallel_function_caller, rho_init)
        opt_rho = results.x
        opt_p = -results.fun

        time_elapsed = time.time() - t0
        print("The optimal rho is", opt_rho)
        print("The period is", opt_p)
        print("Computation Time:", time_elapsed)
def main():
    minimizer(n_runs=1000)

if __name__ == '__main__':
    main()