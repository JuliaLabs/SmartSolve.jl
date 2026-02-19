using LinearSolve
using LinearAlgebra
using BandedMatrices

struct SLUIRFactorization <: LinearSolve.SciMLLinearSolveAlgorithm
    iterations::Int
    function SLUIRFactorization(; iterations = 5)
        return new(iterations)
    end
end

const lwp = Float32

mutable struct SLUIRCache{T}
    n::Int
    r64::Vector{Float64}
    worklwp::Vector{T}
    bflwp::Vector{T}
    x::Vector{Float64}
    Flwp::Any
end

function LinearSolve.init_cacheval(
        alg::SLUIRFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{Bool, LinearSolve.LinearVerbosity}, assump::LinearSolve.OperatorAssumptions)
    return SLUIRCache{lwp}(0, Float64[], lwp[], lwp[], Float64[], nothing)
end

# Fast banded copy with eltype conversion (preserves bandwidths, avoids dense conversion)
@inline function banded_convert(::Type{T}, A::BandedMatrix) where {T}
    B = similar(A, T)
    bdA = BandedMatrices.bandeddata(A)
    bdB = BandedMatrices.bandeddata(B)
    @inbounds for j in axes(bdA, 2), i in axes(bdA, 1)
        bdB[i, j] = T(bdA[i, j])
    end
    return B
end

# Factorize in low precision using BandedMatrices' LU
@inline function banded_factor_lwp(A::BandedMatrix, ::Type{T}) where {T}
    A_T = banded_convert(T, A)
    return lu(A_T)  # banded LU factorization object
end

# Overwrite rhs with solution (low-precision triangular solves via factorization)
@inline function gb_solve!(F, rhs::StridedVector{T}) where {T}
    ldiv!(F, rhs)   # rhs := F \ rhs
    return rhs
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::SLUIRFactorization; kwargs...)
    if cache.isfresh
        n = length(cache.b)
        cache.cacheval.n       = n
        cache.cacheval.r64     = Vector{Float64}(undef, n)
        cache.cacheval.worklwp = Vector{lwp}(undef, n)
        cache.cacheval.bflwp   = Vector{lwp}(undef, n)
        cache.cacheval.x       = Vector{Float64}(undef, n)

        # Low-precision banded LU (Float32) cached once
        cache.cacheval.Flwp = banded_factor_lwp(cache.A, lwp)

        cache.isfresh = false
    end

    n       = cache.cacheval.n
    r64     = cache.cacheval.r64
    worklwp = cache.cacheval.worklwp
    bflwp   = cache.cacheval.bflwp
    x       = cache.cacheval.x
    F       = cache.cacheval.Flwp

    b = cache.b

    # initial solve (Float32) into bflwp, then convert into x (Float64)
    @inbounds for i = 1:n
        bflwp[i] = lwp(b[i])
    end
    gb_solve!(F, bflwp)
    @inbounds for i = 1:n
        x[i] = Float64(bflwp[i])
    end

    # refinement
    for _ = 1:alg.iterations
        mul!(r64, cache.A, x)
        @inbounds for i = 1:n
            r64[i] = b[i] - r64[i]
            worklwp[i] = lwp(r64[i])
        end
        gb_solve!(F, worklwp)
        @inbounds for i = 1:n
            x[i] += Float64(worklwp[i])
        end
    end

    SciMLBase.build_linear_solution(alg, x, nothing, cache)
end
