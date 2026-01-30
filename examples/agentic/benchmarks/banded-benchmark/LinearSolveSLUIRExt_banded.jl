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
const BlasInt = LinearAlgebra.BlasInt

struct GBFactor{T}
    AB::Matrix{T}
    ipiv::Vector{BlasInt}
    n::Int
    kl::Int
    ku::Int
end

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

@inline function gb_pack_and_factor!(A::BandedMatrix, ::Type{T}) where {T}
    n = size(A, 1)
    kl, ku = bandwidths(A)
    ldab = 2*kl + ku + 1

    AB = zeros(T, ldab, n)
    bd = BandedMatrices.bandeddata(A)  # (kl+ku+1)×n

    # pack: copy bd into AB shifted down by kl rows
    @inbounds for j in 1:n
        for r in 1:(kl + ku + 1)
            AB[kl + r, j] = T(bd[r, j])
        end
    end

    AB, ipiv = LAPACK.gbtrf!(kl, ku, n, AB)   # <-- ONLY TWO RETURNS on your Julia
    return GBFactor{T}(AB, ipiv, n, kl, ku)
end

@inline function gb_solve!(F::GBFactor{T}, rhs::StridedVector{T}) where {T}
    LAPACK.gbtrs!('N', F.kl, F.ku, F.n, F.AB, F.ipiv, rhs)  # rhs := solution
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

        cache.cacheval.Flwp = gb_pack_and_factor!(cache.A, lwp)  # cached gbtrf!
        cache.isfresh = false
    end

    n       = cache.cacheval.n
    r64     = cache.cacheval.r64
    worklwp = cache.cacheval.worklwp
    bflwp   = cache.cacheval.bflwp
    x       = cache.cacheval.x
    F       = cache.cacheval.Flwp::GBFactor{lwp}

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
