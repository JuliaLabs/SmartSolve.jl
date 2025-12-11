using SmartSolve
using LinearAlgebra
using SparseArrays
using CUDA
using BenchmarkTools
using Dagger

prompt = """
- Task: Write a high-performance Julia implementation of a linear solver for sparse matrices, using Dagger.jl on the GPU. The solver must be based on Cholesky factorization with iterative refinement.

- Requirements

1) Libraries and references

1.1) Use Dagger.jl as documented here:
https://juliaparallel.org/Dagger.jl/dev/

1.2) Follow the iterative refinement algorithm described here:
https://nhigham.com/2023/03/13/what-is-iterative-refinement/

2) Dagger.jl + Cholesky

2.1) Dagger.jl already has an Cholesky routine that can be used for distributed linear solves: cholesky(A_d) where A_d is a distributed matrix (DMatrix). This implementation extends cholesky from LinearAlgebra.jl. See https://github.com/JuliaParallel/Dagger.jl/blob/master/src/array/cholesky.jl.

2.2) Use that Cholesky implementation within Dagger.jl (do not re-implement Cholesky from scratch).

2.3) The computation must be fully on GPU, using Dagger's GPU support.

2.4) Do not move data back to the CPU for intermediate computations.

2.5) All linear algebra operations (factorization, forward/back substitution, residual computation, refinement updates) must be performed on GPU-resident Dagger arrays.

3) Function API

3.1) Implement exactly one Julia function with the following signature:
function proposed_fn(A_d, b_d)
    # your code here
end
A_d: distributed sparse matrix (Dagger-distributed, GPU-resident).
b_d: distributed vector (Dagger-distributed, GPU-resident).
x: solution vector (Dagger-distributed, GPU-resident). You may treat x as an initial guess and overwrite it with the final refined solution.

3.2) The function must return the final solution x (and anything else you consider useful, e.g., a residual norm, but the first return value must be the solution).

4)Iterative refinement details

4.1) Use Cholesky factorization of A_d to compute an initial solution x₀. 

4.2) Then apply iterative refinement:

    4.2.1) At each iteration k, compute residual r_k = b_d - A_d * x_k on the GPU.
    4.2.2) Solve Ad d_k = r_k using the Cholesky factors (on GPU).
    4.2.3) Update x_k+1 = x_k + d_k on GPU.

4.3) Perform at least 5 refinement iterations (you can use a loop with a fixed number of iterations ≥ 5; optional extra stopping criteria are allowed but not required).

5) Performance and style constraints

5.1) Use Dagger tasks / computation graphs appropriately so that the Cholesky factorization and solves are executed in parallel where possible.

5.2) Avoid unnecessary data movement or conversions.

5.3) Do not use CPU-only arrays or operations (no Array, no collect to CPU, etc.).

5.4) Assume that using LinearAlgebra, using SparseArrays, and using Dagger have already been executed.

5.5) Focus on clarity and correctness first, but structure the code with performance in mind (e.g., reuse LU factors, avoid recomputing them each iteration).

6) Output format

6.1) Output only the Julia code for the function:

function proposed_fn(A_d, b_d)
    ...
end

6.2) Do not include any explanation, comments, or text outside the function definition.

"""

secret_key = ENV["OPENAI_API_KEY"]
solver, hist, conv = gen_linear_solver_dagger(prompt, secret_key; max_iters = 50)

println("Generated Code:\n")
println(solver)
write("solver.jl", solver)




# using Dagger, CUDA, LinearAlgebra
# N = 2000
# A = rand(N, N)
# A = A * A'
# A[diagind(A)] .+= size(A, 1)
# A_d = Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
#     view(A, Blocks(500, 500))
# end
# b_d = Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
#     randn(Blocks(500), N)
# end
# cholesky(A_d) \ b_d