import numpy as np
import pyopencl as cl
import time
import pyopencl.clrandom as clrand
import pyopencl.array as cl_array


def grid_search(n_runs):
    # Set up OpenCL context and command queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    t0 = time.time()

    T = int(4160)  # Set the number of periods for each simulation
    rand_gen = clrand.PhiloxGenerator(ctx)
    ran = rand_gen.normal(queue, (n_runs * T), mu=0, sigma=1)
    rho_l = np.linspace(-0.95, 0.95, 200)

    opt = []
    scan_sim = cl.Program(ctx, """
          __kernel void grid_search(__global float *ary_a, __global float *ary_b, float rho, __global float *result)
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

    result = cl_array.to_device(queue, np.empty(n_runs))
    ary_b = cl_array.to_device(queue, np.empty(n_runs * T))
    for r in rho_l:
        scan_sim.grid_search(queue, (n_runs,), None, ran.data, ary_b.data, r, result.data)
        opt.append(result.get().mean())

    time_elapsed = time.time() - t0
    opt_rho = rho_l[np.argmax(opt)]

    print("The optimal rho is", opt_rho)
    print("The period is", max(opt))
    print("Computation Time:", time_elapsed)

    return


def main():
    grid_search(n_runs = 1000)

if __name__ == '__main__':
    main()