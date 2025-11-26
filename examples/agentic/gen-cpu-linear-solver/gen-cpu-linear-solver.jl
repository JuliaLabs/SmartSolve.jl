using SmartSolve
using LinearAlgebra
using SparseArrays
using BenchmarkTools

prompt = """
Generate a high performance Julia CPU implementation of a linear solver for sparse matrices
ased on LU with iterative refinement (at least 10 iterations of refinement) using the
following reference: https://nhigham.com/2023/03/13/what-is-iterative-refinement
"""

secret_key = ENV["OPENAI_API_KEY"]
code, hist, conv = gen_linear_solver(prompt, secret_key, max_iters = 3)

println("Generated Code:\n")
println(code)
write("generated-cpu-code.jl", code)

println("\nConversation History:\n")
printhist(hist)

println("Simple benchmark:\n")
eval(Meta.parse(code))
A = sprand(10000, 10000, 0.1)
b = rand(10000)
display(@benchmark $A \ $b seconds=20)
display(@benchmark proposed_fn($A, $b) seconds=10)
x1 = A \ b
e1 = norm(A * x1 - b)
x2 = proposed_fn(A, b)
e2 = norm(A * x2 - b)
println("Error of default solver: $e1")
println("Error of generated solver: $e2")
