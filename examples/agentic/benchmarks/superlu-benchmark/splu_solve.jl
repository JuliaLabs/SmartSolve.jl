using SuperLU
using SparseArrays
using LinearAlgebra

"""
    superlu_solver(A, b) -> x

Solve `A * x = b` using SuperLU.jl.

- `A` must be a sparse matrix (CSC preferred).
- `b` can be a vector or a dense matrix (multiple RHS).
Returns `x` with the same shape as `b`.
"""
function superlu_solver(A::SparseMatrixCSC, b::AbstractVecOrMat)
    F = SuperLU.splu(A)
    return F \ b
end
