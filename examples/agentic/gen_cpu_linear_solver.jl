using SmartSolve
using SparseArrays
using BenchmarkTools

prompt = """
Generate a high performance Julia CPU implementation of a linear solver 
for sparse matrices based on LU with iterative refinement using the following reference:
https://nhigham.com/2023/03/13/what-is-iterative-refinement
"""
secret_key = ENV["OPENAI_API_KEY"]
code, hist, conv = gen_linear_solver(prompt, secret_key)

println("Generated Code:\n")
println(code)

println("Simple benchmark:\n")
eval(Meta.parse(code))
A = sprand(10000, 10000, 0.1)
b = rand(10000)
@benchmark $A \ $b
@benchmark proposed_fn(A, b)



