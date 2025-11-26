using SmartSolve
using LinearAlgebra
using SparseArrays
using CUDA
using BenchmarkTools

prompt = """
Generate a high performance Julia CUDA implementation of a linear solver 
for sparse matrices based on LU with iterative refinement using the following reference:
https://nhigham.com/2023/03/13/what-is-iterative-refinement
"""
secret_key = ENV["OPENAI_API_KEY"]
code, hist, conv = gen_linear_solver_cuda(prompt, secret_key)

println("Generated Code:\n")
println(code)
write("generated_cuda_code.jl", code)

println("\nConversation History:\n")
printhist(hist)
