# Discussion on Benchmarks and To Dos

## The benchmarks
This folder contains different benchmarks for Smartsolve solvers. Currently, there are two types of benchmarks.
1. **basic-luir-benchmark**: This folder contains a benchmark of a Smartsolve generated solver, with the caching removed. The intent is to check that the baseline of the solver is functioning correctly and quickly. 
2. **solvername-benchmark**: These folders contain different variations of the SmartSolve generated solver fitted to LinearSolve.jl, using different LU solvers (specified in the folder name) and applying mixed precision iterative refinement.
   
    File description:

    - *LinearSolveSLUIRExt_solvername.jl:* The backend solver
    - *basesolvername_solve.jl:*  The baseline LU solver *without* the mixed precision iterative refinement.
    - *solver_solvername.jl:* The frontend that is called
    - *sluir_benchmark_banded.jl:* Performs the benchmarking
    - *error_vs_time_solvername.pdf:* Results of the benchmark, run on Apple M1 Pro CPU.

## Current Results
- Base solver: matches the performance of the dense mixed precision solver implemented in LAPACK (dsgesv). It achieves better accuracy than dsgesv, but this is likely due to using more steps of iterative definement.
- Banded: Outperforms the baseline solver, but is outperformed by the LAPACK builtin solver. This may be due to matrix size not being large enough.
- MUMPS, Sparspak, and SuperLU: These all performs worse than the built-in backslash solver. They mostly match the performance of the baseline package solver. MUMPS fails on ill conditioned matrices.

## Possible issues with the Benchmarks
- The benchmarks were run on a local machine, not on a cluster. This may lead to errors in the benchmark due to the background processes. However, since each benchmark takes more than one second, it should not be a big issue.
- The dimension and size of matrices: These benchmarks were performed on matrices of dimension of about $10^5 \times 10^5$ for sparse and banded formats. The number of non-zero elements were set to be as large as possible, up to the amount my computer's memory could handle. However, mixed precision iterative refinement solvers work best when the cost of the factorization is as large as possible. 
- Issues with implementation: There may be possible issues with implementations, as these are initial drafts.
- Issues with packages: Some of these packages are quite old and may not be well optimized for current LAPACK/UMFPACK, or Julia versions. The packages may not work well on Apple CPUs. 

## Next Steps
1. Repeat benchmarks at different CPUs, particularly in a cluster.
2. Use matrices of $10^9$ to $10^{11}$ rows and columns and around $10^{11}$ to $10^{13}$ nonzero elements. Or possibly matrices from [real problems](https://sparse.tamu.edu/).
3. Check implementations.
4. Find other packages or methods to implement sparse (or banded) Float32 LU solvers.
5. If 4. does not exists, implement one at a low level using SmartSolve.
6. Find a way to add parallelization explicitly into the solver, rather than the backend solver.