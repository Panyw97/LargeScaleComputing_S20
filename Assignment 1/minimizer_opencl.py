import numpy as np
import pyopencl as cl
import time
import scipy.optimize as opt
import pyopencl.clrandom as clrand
import pyopencl.array as cl_array


def minimizer(rho, args):
    n_runs = args
    # Set up OpenCL context and command queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)


    T = int(4160)  # Set the number of periods for each simulation
    rand_gen = clrand.PhiloxGenerator(ctx)
    ran = rand_gen.normal(queue, (n_runs * T), np.float32, mu=0, sigma=1)

    scan_sim = cl.Program(ctx, """
          __kernel void parallel_compute(__global float *ary_a, __global float *ary_b, 
          float rho, __global float *result)
        {
          int idx = get_global_id(0);
          for (int i=0; i<4160; i++)
          {
            if (i == 0){
              ary_b[idx * 4160 + i] = ary_a[idx * 4160 + i] + 3;
            }
            else {
              ary_b[idx * 4160 + i] = rho * ary_b[idx * 4160 + i - 1] + 3 * (1 - rho) + ary_a[idx * 4160 + i];
            }
            if (ary_b[idx * 4160 + i] <= 0 || i == 4159) {
              result[idx] = i;
              break;
            } 
          }
        }
          """).build()

    result = cl_array.to_device(queue, np.empty(n_runs).astype(np.float32))
    ary_b = cl_array.to_device(queue, np.empty(n_runs * T).astype(np.float32))
    scan_sim.parallel_compute(queue, (n_runs,), None, ran.data, ary_b.data, np.float32(rho), result.data)
    return -(result.get().mean())

    
    opt_rho = rho_l[np.argmax(opt)]

    print("The optimal rho is", opt_rho)
    print("The period is", max(opt))
    print("Computation Time:", time_elapsed)



def main():
    t0 = time.time()
    rho_init = 0.1
    res = opt.minimize(minimizer, rho_init, args=(1000))
    time_elapsed = time.time() - t0
    opt_rho = res.x
    opt_p = -res.fun
    print("The optimal rho is", opt_rho)
    print("The period is", opt_p)
    print("Computation Time:", time_elapsed)

if __name__ == '__main__':
    main()