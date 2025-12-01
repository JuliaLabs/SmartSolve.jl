using SmartSolve
using LinearAlgebra
using SparseArrays
using BenchmarkTools

prompt = """
Generate a high-performance CPU implementation in Julia of a linear solver for sparse matrices
based on LU with iterative refinement (at least 5 refinement iterations), using the following
reference: https://nhigham.com/2023/03/13/what-is-iterative-refinement
"""

secret_key = ENV["OPENAI_API_KEY"]
solver, hist, conv = gen_linear_solver(prompt, secret_key, max_iters = 5)

println("Generated Code:\n")
println(code)
write("solver.jl", solver)