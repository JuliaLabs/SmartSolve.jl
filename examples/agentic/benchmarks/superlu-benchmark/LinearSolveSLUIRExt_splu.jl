using LinearSolve
using SparseArrays
using LinearAlgebra
using SuperLU

struct SLUIRFactorization <: LinearSolve.SciMLLinearSolveAlgorithm
    iterations::Int
    function SLUIRFactorization(; iterations = 5)
        return new(iterations)
    end
end

const lwp = Float32

mutable struct SLUIRCache
    n::Int
    b64::Vector{Float64}       # RHS buffer (double)
    r64::Vector{Float64}       # residual buffer (double)
    bflwp::Vector{lwp}         # RHS buffer (single)
    worklwp::Vector{lwp}       # residual buffer (single)
    x::Vector{Float64}         # solution accumulator (double)
    xflwp::Vector{lwp}         # solution workspace (single)
    dlwp::Vector{lwp}          # correction workspace (single)
    Flwp::Any                  # SuperLU factorization (single)
end

function LinearSolve.init_cacheval(
    alg::SLUIRFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
    verbose::Union{Bool, LinearSolve.LinearVerbosity}, assump::LinearSolve.OperatorAssumptions
)
    return SLUIRCache(0, Float64[], Float64[], lwp[], lwp[], Float64[], lwp[], lwp[], nothing)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::SLUIRFactorization; kwargs...)
    if cache.isfresh
        n = length(cache.b)
        cv = cache.cacheval
        cv.n       = n
        cv.b64     = Vector{Float64}(undef, n)
        cv.r64     = Vector{Float64}(undef, n)
        cv.bflwp   = Vector{lwp}(undef, n)
        cv.worklwp = Vector{lwp}(undef, n)
        cv.x       = Vector{Float64}(undef, n)
        cv.xflwp   = Vector{lwp}(undef, n)
        cv.dlwp    = Vector{lwp}(undef, n)

        # Build Float32 A once (same structure)
        Aorig = cache.A
        Af = SparseMatrixCSC{lwp, Int}(
            size(Aorig, 1), size(Aorig, 2),
            copy(Aorig.colptr), copy(Aorig.rowval),
            lwp.(Aorig.nzval),
        )

        cv.Flwp = SuperLU.splu(Af)

        cache.isfresh = false
    end

    cv = cache.cacheval
    n  = cv.n
    F  = cv.Flwp

    # b64 .= cache.b (no allocations)
    @inbounds @simd for i = 1:n
        cv.b64[i] = cache.b[i]
    end

    # Initial solve (single): xflwp = F \ bflwp, done in-place via ldiv!
    @inbounds @simd for i = 1:n
        bi = lwp(cv.b64[i])
        cv.bflwp[i] = bi
        cv.xflwp[i] = bi
    end
    ldiv!(F, cv.xflwp)

    # x (double) from xflwp
    @inbounds @simd for i = 1:n
        cv.x[i] = Float64(cv.xflwp[i])
    end

    # Iterative refinement
    for _ = 1:alg.iterations
        mul!(cv.r64, cache.A, cv.x)   # r64 = A*x

        @inbounds @simd for i = 1:n
            ri = cv.b64[i] - cv.r64[i]
            cv.r64[i]     = ri
            wi            = lwp(ri)
            cv.worklwp[i] = wi
            cv.dlwp[i]    = wi
        end

        ldiv!(F, cv.dlwp)

        @inbounds @simd for i = 1:n
            cv.x[i] += Float64(cv.dlwp[i])
        end
    end

    return SciMLBase.build_linear_solution(alg, cv.x, nothing, cache)
end
