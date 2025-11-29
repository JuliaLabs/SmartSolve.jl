using SmartSolve
using LinearAlgebra
using SparseArrays
using CUDA
using BenchmarkTools

prompt = """
Generate a high performance Julia CUDA implementation of a linear solver for sparse matrices
ased on LU with iterative refinement (at least 5 iterations of refinement) using the
following reference: https://nhigham.com/2023/03/13/what-is-iterative-refinement
"""

secret_key = ENV["OPENAI_API_KEY"]
code, hist, conv = gen_linear_solver_cuda(prompt, secret_key, max_iters = 5)

println("Generated Code:\n")
println(code)
write("generated-cuda-code.jl", code)


println("Simple GPU benchmark:\n")
eval(Meta.parse(code))
# Problem setup on CPU (sparse)
N = 10_000
A = sprand(N, N, 0.3)
b = rand(N)
Ad = CuArray(Matrix(A))
bd = CuArray(b)

# --- warm-up (compile kernels, allocate, etc.) ---
x1_d = Ad \ bd
x2_d = proposed_fn(Ad, bd)
CUDA.synchronize()

println("GPU benchmark (default solver):")
display(@benchmark begin
    x = $Ad \ $bd
    CUDA.synchronize()
end seconds = 10)

println("\nGPU benchmark (generated solver):")
display(@benchmark begin
    x = proposed_fn($Ad, $bd)
    CUDA.synchronize()
end seconds = 10)

# --- error computation on GPU ---
x1 = Ad \ bd
x2 = proposed_fn(Ad, bd)
CUDA.synchronize()

e1 = norm(Ad * x1 - bd)
e2 = norm(Ad * x2 - bd)

println("\nError of default GPU solver: $e1")
println("Error of generated GPU solver: $e2")