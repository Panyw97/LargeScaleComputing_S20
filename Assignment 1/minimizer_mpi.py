import numpy as np
import scipy.stats as sts
import scipy.optimize as opt
import time
from mpi4py import MPI

# Get rank of process and overall size of communicator:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parallel_function_caller(rho, stopp):
    stopp[0] = comm.bcast(stopp[0], root=0)
    res = 0
    if stopp[0] == 0:
        # Evenly distribute number of simulation runs across processes
        n_runs = 1000
        N = int(n_runs / size)

        # Set model parameters
        mu = 3.0
        sigma = 1.0
        z_0 = mu
        # Set simulation parameters, draw all idiosyncratic random shocks,
        # and create empty containers
        T = int(4160)  # Set the number of periods for each simulation np.random.seed(25)

        if rank == 0:
            eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(N, T))
        else:
            eps_mat = np.empty((N, T))
        comm.Bcast(eps_mat, root=0)
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
        sendbuf = np.array(p)

        # Gather all simulation arrays to buffer of expected size/dtype on rank 0
        recvbuf = None
        if rank == 0:
            recvbuf = np.empty([N * size, 1], dtype='int')
        comm.Gather(sendbuf, recvbuf, root=0)

        if rank == 0:
            # print("call result: ", -np.mean(recvbuf))
            res = -np.mean(recvbuf)
    return res

if __name__ == '__main__':
    if rank == 0:
        stop = [0]
        t0 = time.time()
        rho_init = 0.1
        results = opt.minimize(parallel_function_caller, rho_init, args=(stop,))
        opt_rho = results.x
        opt_p = -results.fun

        time_elapsed = time.time() - t0
        print("The optimal rho is", opt_rho[0])
        print("The period is", opt_p)
        print("Computation Time:", time_elapsed)
        stop = [1]
        parallel_function_caller(rho_init, stop)
    else:
        stop=[0]
        rho_init = 0.1
        while stop[0]==0:
            parallel_function_caller(rho_init, stop)