using LinearSolve
using SparseArrays
using LinearAlgebra
using SciMLBase
using MUMPS
using MPI

struct SLUIRFactorization <: LinearSolve.SciMLLinearSolveAlgorithm
    iterations::Int
    function SLUIRFactorization(; iterations = 5)
        return new{}(iterations)
    end
end

const lwp = Float32

mutable struct SLUIRCache
    n::Int
    r64::Vector{Float64}
    b64buf::Vector{Float64}
    rhs_lwp::Vector{lwp}
    sol_lwp::Vector{lwp}
    cor_lwp::Vector{lwp}
    x::Vector{Float64}
    Alwp::Any
    m::Any                # MUMPS.Mumps{lwp}
end

function LinearSolve.init_cacheval(
    alg::SLUIRFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
    verbose::Union{Bool, LinearSolve.LinearVerbosity}, assump::LinearSolve.OperatorAssumptions)

    return SLUIRCache(0, Float64[], Float64[], lwp[], lwp[], lwp[], Float64[], nothing, nothing)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::SLUIRFactorization; kwargs...)
    if cache.isfresh
        n = length(cache.b)
        cache.cacheval.n      = n
        cache.cacheval.r64    = Vector{Float64}(undef, n)
        cache.cacheval.b64buf = Vector{Float64}(undef, n)

        cache.cacheval.rhs_lwp = Vector{lwp}(undef, n)
        cache.cacheval.sol_lwp = Vector{lwp}(undef, n)
        cache.cacheval.cor_lwp = Vector{lwp}(undef, n)
        cache.cacheval.x       = Vector{Float64}(undef, n)

        # Low-precision copy of A (same as your Sparspak path)
        nzlwp = lwp.(cache.A.nzval)
        Alwp = SparseMatrixCSC{lwp, Int}(size(cache.A,1), size(cache.A,2),
                                         copy(cache.A.colptr), copy(cache.A.rowval),
                                         nzlwp)
        cache.cacheval.Alwp = Alwp

        # MUMPS "default safeties": use default control arrays and the documented constructor
        # and force UNSYMMETRIC mode. :contentReference[oaicite:1]{index=1}
        icntl = copy(MUMPS.default_icntl)

        # suppress printing
        icntl[2] = 0
        icntl[3] = 0
        icntl[4] = 0

        m = MUMPS.Mumps{lwp}(MUMPS.mumps_unsymmetric, icntl, MUMPS.default_cntl32)


        # Default associate_matrix! copies/converts as needed (no unsafe overrides). :contentReference[oaicite:2]{index=2}
        MUMPS.associate_matrix!(m, Alwp)
        MUMPS.factorize!(m)

        cache.cacheval.m = m
        cache.isfresh = false
    end

    n       = cache.cacheval.n
    r64     = cache.cacheval.r64
    b64buf  = cache.cacheval.b64buf
    rhs_lwp = cache.cacheval.rhs_lwp
    sol_lwp = cache.cacheval.sol_lwp
    cor_lwp = cache.cacheval.cor_lwp
    x       = cache.cacheval.x
    m       = cache.cacheval.m

    # b64 view/buffer (avoid alloc)
    b = cache.b
    b64 = if eltype(b) === Float64
        b
    else
        @inbounds for i = 1:n
            b64buf[i] = Float64(b[i])
        end
        b64buf
    end

    # Initial solve (low precision)
    @inbounds for i = 1:n
        rhs_lwp[i] = lwp(b64[i])
    end
    MUMPS.associate_rhs!(m, rhs_lwp)   # default behavior (no unsafe override) :contentReference[oaicite:3]{index=3}
    MUMPS.solve!(m)
    MUMPS.get_sol!(sol_lwp, m)         # retrieve solution from MUMPS object :contentReference[oaicite:4]{index=4}

    @inbounds for i = 1:n
        x[i] = Float64(sol_lwp[i])
    end

    # Iterative refinement (your loop)
    for _ = 1:alg.iterations
        mul!(r64, cache.A, x)  # r64 = A*x
        @inbounds for i = 1:n
            rhs_lwp[i] = lwp(b64[i] - r64[i])  # residual
        end
        MUMPS.associate_rhs!(m, rhs_lwp)
        MUMPS.solve!(m)
        MUMPS.get_sol!(cor_lwp, m)

        @inbounds for i = 1:n
            x[i] += Float64(cor_lwp[i])
        end
    end

    return SciMLBase.build_linear_solution(alg, x, nothing, cache)
end
