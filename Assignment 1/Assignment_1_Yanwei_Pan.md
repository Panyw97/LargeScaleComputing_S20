# Assignment #1



## Yanwei Pan



### 1. Clocking CPU parallelism

(a)  Simulation Code: [clock.py](https://github.com/Panyw97/LargeScaleComputing_S20/blob/master/Assignment%201/clock.py)  
  Plot Code: [Assignment1_Q1(a).ipynb](https://colab.research.google.com/github/jonclindaniel/LargeScaleComputing_S20/blob/master/Labs/Lab%202%20PyOpenCL/Lab_2_PyOpenCL_Random_Walk_Tutorial.ipynb#scrollTo=lzORBsZvrhVt)
  ![plot](https://github.com/Panyw97/LargeScaleComputing_S20/blob/master/Assignment%201/health_simulation_mpi.png)

(b) First of all, according to Amdahl's law, the speedup of N-cores parallel computing is not only up to the number of cores (processors) we use but also up to the fraction of parallel and serial code. Therefore, the speedup isn't linear. Also, there are two bottlenecks in the multicore CPU computing process: the computing bottleneck and the memory bottleneck. In this simulation task, the computing bottleneck does exist. As a result, as the number of cores increases, the speedup couldn't increase linearly.  

### 2. Clocking GPU parallelism

(a) Simulation Code: [gpu.py](https://github.com/Panyw97/LargeScaleComputing_S20/blob/master/Assignment%201/gpu.py)  
It takes 2.310169219970703s to run.

(b) This is quite faster than the mpi4py implementation for 1 core. However, it's still slower than the multicore implementation. This is because we only use one GPU. Compared with CPU, GPU can run much faster for computation.

### 3. Embarrassingly parallel processing: grid search

(a) Code: [grid_search_mpi.py](https://github.com/Panyw97/LargeScaleComputing_S20/blob/master/Assignment%201/grid_search_mpi.py)  
Computation Time for MPI: 11.182556629180908 

(b) Code: [grid_search_opencl.py](https://github.com/Panyw97/LargeScaleComputing_S20/blob/master/Assignment%201/grid_search_opencl.py)  
Computation Time for OpenCL: 0.12177801132202148  
The OpenCL implementation is much faster than MPI. This is because GPU is good at computing. As this task involves computing the average period of each rho, the advantage of using GPU is more explicit.

(c) ![plot](https://github.com/Panyw97/LargeScaleComputing_S20/blob/master/Assignment%201/Avg%20Periods%20VS%20Rho.png)

(d)    
The results from mpi4py implementation:  
    The optimal rho is 0.03341708542713562  
    The period is 754.707  
  
The results from PyOpenCL implementation:  
    The optimal rho is 0.004773869  
    The period is 727.259  

### 4. More sophisticated parallelism: minimizer

(a)  Code: [minimizer_mpi.py](https://github.com/Panyw97/LargeScaleComputing_S20/blob/master/Assignment%201/minimizer_mpi.py)  
Computation Time for MPI: 20.86929202079773    

(b)  Code: [minimizer_opencl.py](https://github.com/Panyw97/LargeScaleComputing_S20/blob/master/Assignment%201/minimizer_opencl.py)  
Computation Time for OpenCL: 15.51089358329773  

(c)  
The results from mpi4py implementation:  
    The optimal rho is 0.10005436756323129  
    The period is 2611.584  
    (However, as I ran the same code on Colab, the result is:  
    The optimal rho is 0.09992672365444555  
    The period is 716.863  
    which seems to be the correct answer. But I haven't debugged it successfully.)  

The results from PyOpenCL implementation:  
    The optimal rho is 0.09998509  
    The period is 698.0989990234375  

(d) This exercise is quite slower than the grid search implementations in Excercise 3. That is because `scipy.optimize.minimize()` is a serial function. Although we used parallel computing as we scattered the simulations to every core or kernel, we still need to change the value of `rho` serially. As there may be lots of rhos that we need to try in `scipy.optimize.minimize()`, the computation time may be quite longer than the grid search, which we could previously know the amount of rhos we need to try.




