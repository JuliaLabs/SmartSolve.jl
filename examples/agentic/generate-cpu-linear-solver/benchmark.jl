using LinearAlgebra
using SparseArrays
using BenchmarkTools

include("solver.jl")

println("Simple CPU benchmark:\n")
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
