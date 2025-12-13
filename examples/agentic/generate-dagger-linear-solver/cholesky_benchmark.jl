using LinearAlgebra, CUDA, Dagger, BenchmarkTools

# SPD Matrix
N = 10_000
A = randn(N, N)
A = A*A' + N*I

# Right-hand side
b = randn(N)


# Cholesky: CUDA #############################################################

# SPD Matrix and right-hand side on GPU (CUDA)
A_cuda = CUDA.CuArray(A)
b_cuda = CUDA.CuArray(b)

# Warm-up
x_cuda = cholesky(A_cuda) \ b_cuda

# Benchmark time and memory
@time cholesky(A_cuda) #  0.895192 seconds (660 allocations: 12.094 KiB)
#@benchmark cholesky($A_cuda)
@time cholesky(A_cuda) \ b_cuda  # 0.882263 seconds (953 allocations: 16.844 KiB)
#@benchmark cholesky($A_cuda) \ $b_cuda

# Errors
e_cuda = norm(A_cuda*x_cuda - b_cuda)/norm(b_cuda)

# Free memory
A_cuda = nothing
b_cuda = nothing
GC.gc()
CUDA.reclaim()


# Cholesky: Dagger ########################################################### 

# SPD Matrix and right-hand side on GPU (Dagger Distributed)
A_d = Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
    distribute(A, Blocks(N÷4, N÷4))
end
b_d = Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
    distribute(b, Blocks(N÷4))
end

# Warm-up
x_d = Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
    cholesky(A_d)
end
Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
    cholesky(A_d) \ b_d
end

CUDA.reclaim()

# Benchmark time and memory
@time Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
     cholesky(A_d)
end #  1.039517 seconds (88.80 k allocations: 4.146 MiB, 7.74% gc time, 18 lock conflicts, 5.70% compilation time: <1% of which was recompilation)
@time Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
     cholesky(A_d) \ b_d
end #  1.046506 seconds (88.80 k allocations: 4.146 MiB, 7.74% gc time, 18 lock conflicts, 5.70% compilation time: <1% of which was recompilation)
# @benchmark Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
#     cholesky($A_d)
# end samples = 1
# @benchmark Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
#     cholesky($A_d) \ $b_d
# end samples = 1
# Free memory

# Errors
e_d = norm(A_d*x_d - b_d)

# Free memory
A_d = nothing
b_d = nothing
GC.gc()

