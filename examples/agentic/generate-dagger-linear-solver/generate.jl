using SmartSolve
using LinearAlgebra
using SparseArrays
using CUDA
using BenchmarkTools
using Dagger

prompt = """
Generate a high-performance Dagger.jl (https://juliaparallel.org/Dagger.jl/dev/) implementation in Julia of a linear solver for sparse matrices
based on LU with iterative refinement (at least 5 refinement iterations), using the following
reference: https://nhigham.com/2023/03/13/what-is-iterative-refinement
"""

secret_key = ENV["OPENAI_API_KEY"]
solver, hist, conv = gen_linear_solver_dagger(prompt, secret_key; max_iters = 5)

println("Generated Code:\n")
println(solver)
write("solver.jl", solver)