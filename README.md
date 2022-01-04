# Cuda-Study
Example study for GPU and Cuda programming

* ### SAXPY
SAXPY stands for "Single-Precision AX Plus Y". It is a function in the standard basic linear algebra subroutines library. 
Compared to the sequential CPU-based implementation, SAXPY can parallelly compute N element of each array. 
Host will send N element of xarray and yarray using cudaMalloc and cudaMemcpy. 
Programmer writes a single function that operates over multiple data (SPMD programming model) by calling saxpyCUDA in main.cpp and 512 thread execute the same function. 
As a result, we can expect faster computation of result = scale*X+Y by using SPMD compared to sequential CPU model.

* ### MatrixMultiplication
Computing matrix multiplication has a lot of locality by executing loop row by row and column by column. 
Unfortunately, for the naïve implementation, this locality is not considered. 
Thus the computation-memory ratio is too low and performance is bottlenecked by memory bandwidth. To alleviate this problem, shared-memory based implementation is used. 
Explicitly defined shared memory is used during the thread block and shared-memory based implementation exploit data locality by stashing frequently used data in shared memory. 
Shared memory allows each value to be accessed by multiple threads and improves memory bandwidth utilization via data-reuse. 
As a result, shared-memory based implementation shows almost 2 times better performance than naïve one.

* ### Simple Circle Renderer
