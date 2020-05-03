import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
import time


def sim_life_parallel(n_runs):
    # Set up context and command queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mem_pool = cltools.MemoryPool(cltools.ImmediateAllocator(queue))

    # Start time:
    t0 = time.time()

    # Generate an array of Normal Random Numbers on GPU of length n_sims*n_steps
    T = int(4160) # Set the number of periods for each simulation np.random.seed(25)
    rand_gen = clrand.PhiloxGenerator(ctx)
    ran = rand_gen.normal(queue, (n_runs * T), np.float32, mu=0, sigma=1)


    # GPU: Define Kernel
    scan_sim = cl.Program(ctx, """
        __kernel void sim(__global float *ary)
        {
            int idx = get_global_id(0);
            for (int i = 0; i < 4160; i++ )
            {
            if (i == 0) {
            ary[idx * 4160 + i] = ary[idx * 4160 + i] + 3.0;
            }
            else {
            ary[idx * 4160 + i] = 0.5 * ary[idx * 4160 + i - 1] + 1.5 +
            ary[idx * 4160 + i];
            }
            }
        }""").build()

    scan_sim.sim(queue, (n_runs,), None, ran.data)
    result = (ran.get().reshape(n_runs, T).transpose())
    
    time_elapsed = time.time() - t0
    print(time_elapsed)
    return

def main():
    sim_life_parallel(n_runs = 1000)

if __name__ == '__main__':
    main()