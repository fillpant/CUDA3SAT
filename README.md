# CUDA3SAT
CUDA3SAT is a GPGPU based SAT checker aimed at quickly checking partly explored 3-SAT problems aiming to determine if some value to the last few variables results in a satisfying assignment.

This tool is not intended to act as a full sat solver, but rather part of one used as part of a 'clever' CPU-based solver which is part of our future work (detailed bellow)

## Setup
You will need a machine with the CUDA toolkit (Version >= 4) and one or more NVIDIA GPUs with compute capability >= 5. This code has not been tested on lower compute capabilities.

### Setup steps
 1. Clone this repo
 2. Open the makefile and change the `GPU_ARCHITECTURE` variable to be compatible with the target GPGPUs. Note that on multi-GPU platforms this would have to be the minimum value of those supported by the targets.
 3. Run `make` in the cloned repo directory
 4. Once compiled the executable named `out` should be available. 

Please note, this project is not yet ready to run. It is currently made to attempt and solve 3-CNF instances with <=64 variables completely which is not its intended purpose. 

## Current Tests and Benchmarks
We have used this code to gather some data on how this approach performs on different devices. 

The input data used can be found in this repo under the directory "inputs". 

The bellow table presents the average kernel time across 10 runs (in milliseconds) for each input on each setup of cards.
|Input| 1x1080ti | 2x1080ti | 2080ti |
|--|--|--|--|
|25-200|22.95|13.27|10.46|
|25-400|23.43|11.94|9.73|
|30-200|603.3|306|212.8|
|30-400|613.9|278.9|210.6|
|35-200|19688.7|10479.9|7686.9|
|35-400|20639.3|8416.2|8319.2|
|40-200|680554|324637|240549.1|
|40-400|679337.8|322828.3|241057.1|

## Future Work
We plan to extend this work to both become part of a hybrid solver, and become more performant. More specifically we plan to:
 - Integrate with a CPU side solver running a 'clever' search algorithm
 - Examine different applications in domains such as ALLSAT which inherently involve larger search space explorations
 - Examine potential use of in-warp directives in better coordinating warps
 - Examine impact of gray codes on thread generated local assignments
 - Move part of the sorted expression to the remaining shared memory that is available as to reduce traffic to global memory, and alter the checking code to consider the shared memory part first before performing global memory transaction
 - Reduce register pressure on functions such as `d_check`
 - Examine impact of manual use of `LOP3` for logic operations (or better, compiler hints that result in `LOP3` use)
 
 

