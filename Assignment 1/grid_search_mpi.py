import numpy as np
import scipy.stats as sts
import time
from mpi4py import MPI


def sim_life_parallel(n_runs):
    T = int(4160) # Set the number of periods for each simulation np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, n_runs))

	# Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start time:
    t0 = time.time()

    # Evenly distribute number of simulation runs across processes
    N = int(n_runs/size)

	# Set model parameters
    mu = 3.0
    sigma = 1.0
    z_0 = mu
    rho_l = np.linspace(-0.95, 0.95, 200)
    
    z_mat = np.zeros((T, n_runs))
    z_mat[0, :] = z_0
    p_avg_dict = {}
    for rho_ind in range(N):
        rho = rho_l[rho_ind]
        p = []
        for s_ind in range(n_runs):
            z_tm1 = z_0
            for t_ind in range(T):
                e_t = eps_mat[t_ind, s_ind]
                z_t = rho * z_tm1 + (1 - rho) * mu + e_t
                if z_t <= 0:
                    p.append(t_ind)
                    break
                z_mat[t_ind, s_ind] = z_t
                z_tm1 = z_t
        p_avg = mean(p)
        print(rho, p_avg)
        p_avg_dict[rho] = p_avg
    opt_rho = min(p_avg_dict, key=p_avg_dict.get)
    opt_p = p_avg_dict[opt_rho]

# Gather all simulation arrays to buffer of expected size/dtype on rank 0
    rho_all = None
    if rank == 0:
        rho_all = np.empty([1, size], dtype='float')
    comm.Gather(sendbuf = (opt_rho, opt_p), recvbuf = rho_all, root=0)
    if rank == 0:
        d = {}
        for r, t in rho_all:
            d[r] = t
        opt_result = min(d, key=d.get)
        time_elapsed = time.time() - t0
        print(time_elapsed)
    return

def main():
    sim_life_parallel(n_runs = 1000)

if __name__ == '__main__':
    main()