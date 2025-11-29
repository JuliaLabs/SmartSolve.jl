using LinearAlgebra
using SparseArrays
using CUDA
using BenchmarkTools

println("Simple GPU benchmark:\n")

include("solver.jl")

# Problem setup on CPU (sparse)
N = 10_000
A = sprand(N, N, 0.3)
b = rand(N)
Ad = CuArray(Matrix(A))
bd = CuArray(b)
x2_d = CuArray(zeros(N))

# --- warm-up (compile kernels, allocate, etc.) ---
x1_d = Ad \ bd
CUDA.CUSOLVER.gesv!(x2_d, Ad, bd)
CUDA.synchronize()
bd = CuArray(b) # reset rhs
x3_d = proposed_fn(Ad, bd)
CUDA.synchronize()

println("GPU benchmark (default solver):")
display(@benchmark begin
    x = $Ad \ $bd
    CUDA.synchronize()
end seconds = 10)

println("\nGPU benchmark (gesv! solver):")
display(@benchmark begin
    CUDA.CUSOLVER.gesv!(x2_d, Ad, bd)
    CUDA.synchronize()
end seconds = 10)

bd = CuArray(b)
println("\nGPU benchmark (generated solver):")
display(@benchmark begin
    x = proposed_fn($Ad, $bd)
    CUDA.synchronize()
end seconds = 10)

# --- error computation on GPU ---
x1_d = Ad \ bd
CUDA.CUSOLVER.gesv!(x2_d, Ad, bd)
x3_d = proposed_fn(Ad, bd)
CUDA.synchronize()

e1 = norm(Ad * x1_d - bd)
e2 = norm(Ad * x2_d - bd)
e3 = norm(Ad * x3_d - bd)

println("\nError of default GPU solver: $e1")
println("Error of gesv! GPU solver: $e2")
println("Error of generated GPU solver: $e3")