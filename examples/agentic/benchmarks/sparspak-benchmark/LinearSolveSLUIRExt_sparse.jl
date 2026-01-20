using LinearSolve
using SparseArrays
using LinearAlgebra
using Sparspak

"""
Single/low-precision LU + double-precision iterative refinement, wrapped for LinearSolve.jl.
Set `lwp` to choose the low-precision type used for the Sparspak LU (default Float32).
"""
struct SLUIRFactorization{T<:AbstractFloat} <: LinearSolve.SciMLLinearSolveAlgorithm
    iterations::Int
end

SLUIRFactorization(; iterations::Integer = 5, lwp::Type{T} = Float32) where {T<:AbstractFloat} =
    SLUIRFactorization{T}(Int(iterations))

mutable struct SLUIRCache{T<:AbstractFloat, IT<:Integer}
    n::Int
    r64::Vector{Float64}     # residual buffer (double)
    worklwp::Vector{T}       # residual in low precision
    bflwp::Vector{T}         # rhs in low precision
    x64::Vector{Float64}     # solution in double (persistent)
    xlwp::Vector{T}          # low-precision solution buffer
    dlwp::Vector{T}          # low-precision correction buffer
    Flwp::Any                # Sparspak factorization (kept as Any to avoid tight coupling)
end

function LinearSolve.init_cacheval(
    alg::SLUIRFactorization{T}, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
    verbose::Union{Bool, LinearSolve.LinearVerbosity}, assump::LinearSolve.OperatorAssumptions
) where {T}
    IT = (A isa SparseMatrixCSC) ? eltype(A.rowval) : Int
    return SLUIRCache{T,IT}(0, Float64[], T[], T[], Float64[], T[], T[], nothing)
end

# Try in-place solve if supported; otherwise fall back to allocating "\" and copy into `out`.
@inline function _ldiv_or_backslash!(out, F, rhs)
    
        out .= F \ rhs
    return out
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::SLUIRFactorization{T}; kwargs...) where {T}
    A = cache.A
    if cache.isfresh
        n = length(cache.b)
        cache.cacheval.n = n
        cache.cacheval.r64     = Vector{Float64}(undef, n)
        cache.cacheval.worklwp = Vector{T}(undef, n)
        cache.cacheval.bflwp   = Vector{T}(undef, n)
        cache.cacheval.x64     = Vector{Float64}(undef, n)
        cache.cacheval.xlwp    = Vector{T}(undef, n)
        cache.cacheval.dlwp    = Vector{T}(undef, n)

        IT = eltype(A.rowval)
        Af = SparseMatrixCSC{T,IT}(size(A,1), size(A,2),
                                   copy(A.colptr), copy(A.rowval),
                                   T.(A.nzval))
        cache.cacheval.Flwp = Sparspak.sparspaklu(Af; factorize = true)

        cache.isfresh = false
    end

    n       = cache.cacheval.n
    r64     = cache.cacheval.r64
    worklwp = cache.cacheval.worklwp
    bflwp   = cache.cacheval.bflwp
    x64     = cache.cacheval.x64
    xlwp    = cache.cacheval.xlwp
    dlwp    = cache.cacheval.dlwp
    Flwp    = cache.cacheval.Flwp

    # Initial solve (low precision), accumulate in Float64
    @inbounds for i in 1:n
        bflwp[i] = T(cache.b[i])
    end
    _ldiv_or_backslash!(xlwp, Flwp, bflwp)
    @inbounds for i in 1:n
        x64[i] = Float64(xlwp[i])
    end

    # Iterative refinement: residual in Float64, correction via low-precision LU
    for _ in 1:alg.iterations
        mul!(r64, cache.A, x64)  # r64 = A*x64
        @inbounds for i in 1:n
            ri = Float64(cache.b[i]) - r64[i]
            r64[i] = ri
            worklwp[i] = T(ri)
        end
        _ldiv_or_backslash!(dlwp, Flwp, worklwp)
        @inbounds for i in 1:n
            x64[i] += Float64(dlwp[i])
        end
    end

    return SciMLBase.build_linear_solution(alg, x64, nothing, cache)
end


# A = sprand(Float64, 10_000, 10_000, 1e-4) + I
# b = randn(size(A,1))
# prob = LinearProblem(A, b)
# sol = solve(prob, SLUIRFactorization(iterations=5, lwp=Float32))
# x = sol.u
