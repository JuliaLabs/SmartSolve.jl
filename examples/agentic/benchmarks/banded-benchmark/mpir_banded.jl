using LinearAlgebra
using BandedMatrices
using FillArrays

# ---------------------------
# Utilities
# ---------------------------

"""
    banded_convert(::Type{T}, A::AbstractBandedMatrix) -> BandedMatrix{T}

Convert a banded matrix to element type `T` without densifying, preserving bandwidth.
"""
function banded_convert(::Type{T}, A::BandedMatrices.AbstractBandedMatrix) where {T}
    l, u = BandedMatrices.bandwidths(A)
    m, n = size(A)
    B = BandedMatrix(Zeros(T, m, n), (l, u))

    @inbounds for j in 1:n
        ilo = max(1, j - u)
        ihi = min(m, j + l)
        for i in ilo:ihi
            B[i, j] = T(A[i, j])
        end
    end
    return B
end

"""
    banded_infnorm(A::AbstractBandedMatrix) -> Real

Compute ‖A‖_∞ (max row sum) in O(n*(l+u)) using the band structure.
"""
function banded_infnorm(A::BandedMatrices.AbstractBandedMatrix{T}) where {T}
    l, u = BandedMatrices.bandwidths(A)
    n = size(A, 1)
    @assert size(A, 2) == n
    maxsum = zero(real(T))

    @inbounds for i in 1:n
        s = zero(real(T))
        jlo = max(1, i - l)
        jhi = min(n, i + u)
        for j in jlo:jhi
            s += abs(A[i, j])
        end
        maxsum = max(maxsum, s)
    end
    return maxsum
end

# ---------------------------
# Factorization object
# ---------------------------

mutable struct MPIRFactor{Tl,Th,TA,TF}
    Ahi::TA          # high precision banded matrix
    Flo::TF          # low precision LU factorization of Alo
    nAinf::Th        # ‖A‖∞ in high precision
    Ax::Vector{Th}   # workspace: A*x
    r::Vector{Th}    # workspace: residual
    buf::Vector{Tl}  # workspace: low-prec RHS / correction
end

"""
    mpir_factorize(A; lowT=Float32, highT=Float64)

Prepare mixed-precision iterative refinement for banded A:
- store A in `highT`,
- compute LU(A) in `lowT`,
- allocate work buffers for repeated solves.
"""
function mpir_factorize(A::BandedMatrices.AbstractBandedMatrix;
                        lowT::Type = Float32,
                        highT::Type = Float64)

    Ahi = banded_convert(highT, A)
    Alo = banded_convert(lowT,  Ahi)

    # LU in low precision (banded factorization path)
    Flo = lu(Alo)

    n = size(Ahi, 1)
    return MPIRFactor{lowT, highT, typeof(Ahi), typeof(Flo)}(
        Ahi, Flo, highT(banded_infnorm(Ahi)),
        Vector{highT}(undef, n),
        Vector{highT}(undef, n),
        Vector{lowT}(undef, n),
    )
end

# ---------------------------
# Solve
# ---------------------------

"""
    mpir_solve(F, b; maxiter=10, rtol=10eps(highT), atol=0, x0=nothing, verbose=false)

Solve A*x=b using mixed precision iterative refinement.

Returns (x, info) where info is a NamedTuple:
  (converged, iters, berr, berr_hist)
"""
function mpir_solve(F::MPIRFactor{Tl,Th}, b::AbstractVector;
                    maxiter::Int = 10,
                    rtol = 10*eps(Th),
                    atol = zero(Th),
                    x0 = nothing,
                    verbose::Bool = false) where {Tl,Th}

    n = length(b)
    @assert size(F.Ahi, 1) == n

    # High-precision RHS copy (keeps API simple)
    bhi = Vector{Th}(undef, n)
    @inbounds for i in 1:n
        bhi[i] = Th(b[i])
    end
    bnorm = norm(bhi, Inf)

    # Initial solve: low precision LU backsolve, then cast to high precision
    x = Vector{Th}(undef, n)
    if x0 === nothing
        @inbounds for i in 1:n
            F.buf[i] = Tl(bhi[i])
        end
        ldiv!(F.Flo, F.buf)  # buf := Alo \ b
        @inbounds for i in 1:n
            x[i] = Th(F.buf[i])
        end
    else
        @inbounds for i in 1:n
            x[i] = Th(x0[i])
        end
    end

    berr_hist = Vector{Th}(undef, maxiter + 1)

    # refinement loop
    for k in 0:maxiter
        # r = b - A*x   (computed in high precision)
        mul!(F.Ax, F.Ahi, x)
        @inbounds for i in 1:n
            F.r[i] = bhi[i] - F.Ax[i]
        end

        # relative backward error (Higham's style):
        # ‖r‖∞ / (‖A‖∞‖x‖∞ + ‖b‖∞)
        denom = F.nAinf * norm(x, Inf) + bnorm
        berr = norm(F.r, Inf) / denom
        berr_hist[k+1] = berr

        if verbose
            @info "mpir iter=$k berr=$berr"
        end

        if berr ≤ max(atol, rtol)
            return x, (converged = true, iters = k, berr = berr, berr_hist = berr_hist[1:k+1])
        end

        if k == maxiter
            break
        end

        # Solve correction in low precision: Alo * d = r
        @inbounds for i in 1:n
            F.buf[i] = Tl(F.r[i])
        end
        ldiv!(F.Flo, F.buf)   # buf := d_lo

        # Update in high precision: x += d
        @inbounds @simd for i in 1:n
            x[i] += Th(F.buf[i])
        end
    end

    berr_last = berr_hist[maxiter+1]
    return x, (converged = false, iters = maxiter, berr = berr_last, berr_hist = berr_hist)
end
