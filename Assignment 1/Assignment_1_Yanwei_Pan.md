# Assignment #1



## Yanwei Pan



### 1. Clocking CPU parallelism

(a)  

(b) First of all, according to Amdahl's law, the speedup of N-cores parallel computing is not only up to the number of cores (processors) we use but also up to the fraction of parallel and serial code. Therefore, the speedup isn't linear. In addition, there are two bottlenecks in the multicore CPU computing process: the computing bottleneck and the memory bottleneck. In this simulation task, the computing bottleneck do exist. As a result, as the number of cores increases, the speedup couldn't increase linearly.  

### 2. Clocking GPU parallelism

(a) It takes 2.310169219970703s to run.

(b) This is quite faster than the mpi4py implementation for 1 core. However, it's still slower than the multicore implementation. This is because we only use one GPU. Compared with CPU, GPU can run much faster for computation.

### 3. Embarrassingly parallel processing: grid search

(a) Computation Time for MPI: 11.182556629180908 

(b) Computation Time for OpenCL: 0.12177801132202148  
The OpenCL implementation is quite faster than MPI. This is because GPU is good at computing. As this task involves computing the average period of each rho, the advantage of using GPU is more explicit.

(c) 

(d) 
The results from mpi4py implementation: 
    The optimal rho is 0.03341708542713562
    The period is 754.707

The results from PyOpenCL implementation:  
    The optimal rho is 0.004773869  
    The period is 727.259  

### 4.

Please make sure **you have installed all of the modules mentioned in `requirements.txt`**. 

### Testing code

For testing, please run the following command from the linux command-line:

<p>$ python test.py NUMBER STORE</p>

NUMBER is the number of items you want to get for the sample. Due to the consideration of time, please enter a number smaller tha 10. STORE is the store (or the crawler) you want to test, which has three options:

- TJ (stands for testing Trader Joes Crawler)
- WF (stands for testing Whole Foods Crawler)
- JO (stands for testing Jewel Osco Crawler)



## 2. Examples



This is an example of how you may test the crawlers after clone our repository locally. To get a test data sample for each of the three grocery stores (Trader Joe's, Whole Foods and Jewel Osco), you can run `test.py`follow these simple example steps.

### Testing Trader Joe's Crawler

`$ python test.py 3 TJ`

Then you will get a small sample of data that scraped from Trader Joe's website showing on your linux command-line. 

**--Special Notice:--**

1. Regrettably, you can't set the number of testing items that you want to get for this crawler, because `TraderJoesCrawler` was built based on three approaches, each of which can obtain a part of items in this store. You can only set the NUMBER to be as small as possible (like 1 or 2) in order to get a relatively small size of test sample. But you can't limit the number of items you get from the test.

2. You may get some non-food items in the test result, whose detailed information are `NaN`. This is correct because we will clean and drop these data in the subsequent data cleaning process.

### Testing Whole Foods' Crawler

`$ python test.py 3 WF`

Then you will get 3 pieces of sample data.

### Testing Jewel Osco's Crawler

`$ python test.py 3 JO`

After running this code, you will see following web page, which requires you to log in.

![Alt text](https://user-images.githubusercontent.com/54608538/76725932-b664e700-671d-11ea-938c-e6b043ef3e92.png)

Click **log in** to type in the account

![Alt text](https://user-images.githubusercontent.com/54608538/76725966-d0062e80-671d-11ea-970b-227552a0a02a.png)

Here we provide you a testing account for logging in.

- Email Address: mingtao0123@gmail.com
- Password: 123456

Please make sure you successfully log into the account within 30s after the chrome pop up. Otherwise, there will be an error in the test.

Then you will get 3 pieces of sample data in this test.

**--Special Notice:--**

Testing JOCrawler may take some time. This is beacause we need to get the labels for all of the items in the department first and then get the food item information. Please keep patient while testing JOCrawler! :-P



