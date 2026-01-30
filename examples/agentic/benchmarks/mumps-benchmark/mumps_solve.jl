using MUMPS
using MPI
using SparseArrays
using LinearAlgebra

"""
    mumps_lu_solve_fast(A, b) -> x

Benchmark-only one-shot: LU factorize A with MUMPS and solve A*x=b.

NO finalize calls:
  * does NOT call `MUMPS.finalize!`
  * does NOT call `MPI.Finalize()`

ASSUMES (no checks):
  * MPI.Init() has already been called.
  * A isa SparseMatrixCSC{T,Int} and b isa Vec/Mat{T}.
  * A and b are not modified while MUMPS holds unsafe pointers.

NOTE: printing is suppressed via MUMPS ICNTL.
"""
@inline function mumps_lu_solve_fast(A::SparseMatrixCSC{T,Int}, b::AbstractVecOrMat{T}) where {T}
    # Start from MUMPS default controls (i.e., MUMPS's own "safeties"),
    # but suppress printing.
    icntl = copy(MUMPS.default_icntl)
    icntl[2] = 0   # diagnostics/statistics stream suppressed
    icntl[3] = 0   # global information stream suppressed
    icntl[4] = 0   # print level = 0

    cntl = (T <: Union{Float32,ComplexF32}) ? MUMPS.default_cntl32 : MUMPS.default_cntl64

    m = MUMPS.Mumps{T}(MUMPS.mumps_unsymmetric, icntl, cntl)

    MUMPS.associate_matrix!(m, A; unsafe=true)
    MUMPS.factorize!(m)

    MUMPS.associate_rhs!(m, b; unsafe=true)
    MUMPS.solve!(m)

    return MUMPS.get_solution(m)
end

# MPI.Init()
# A = sprand(Float64, 100_000, 100_000, 5e-6) + I
# b = randn(size(A,1))
# x = mumps_lu_solve_fast(A, b)
