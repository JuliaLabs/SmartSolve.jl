using LinearAlgebra
using SparseArrays
using BenchmarkTools

include("solver.jl")

N = 15_000
A = sprand(N, N, 0.5) 
b = rand(N)

# See https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.lu
umfpack_control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int64) # read Julia default configuration for a Float64 sparse matrix
#SparseArrays.UMFPACK.show_umf_ctrl(umfpack_control) # optional - display values
umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP] = 2.0 # reenable iterative refinement (2 is UMFPACK default max iterative refinement steps)

# warm-up
x1 = A \ b   # solve Ax = b
x2 = lu(A; control = umfpack_control) \ b   # solve Ax = b, including UMFPACK iterative refinement
x3 = proposed_fn(A, b)

@benchmark begin
    x = $A \ $b   # solve Ax = b
end

@benchmark begin
    x = lu($A; control = $umfpack_control) \ $b   # solve Ax = b, including UMFPACK iterative refinement
end

@benchmark begin
    x = proposed_fn($A, $b) 
end

error1 = norm(A*x1 - b) / norm(b)

error2 = norm(A*x2 - b) / norm(b)

error3 = norm(A*x3 - b) / norm(b)