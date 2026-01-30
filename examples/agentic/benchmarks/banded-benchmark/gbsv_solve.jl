using LinearAlgebra
using BandedMatrices

const BlasInt   = LinearAlgebra.BlasInt
const liblapack = LinearAlgebra.LAPACK.liblapack

# Float64
function gbsv_solve(A::BandedMatrix{Float64}, b::StridedVector{Float64})
    n = size(A, 1)
    kl, ku = bandwidths(A)
    ldab = 2*kl + ku + 1

    AB = zeros(Float64, ldab, n)
    bd = BandedMatrices.bandeddata(A)                 # (kl+ku+1)×n
    @inbounds @views AB[(kl+1):ldab, :] .= bd          # pack into LAPACK GB storage

    ipiv = Vector{BlasInt}(undef, n)
    x    = copy(b)                                     # LAPACK overwrites RHS with solution
    info = Ref{BlasInt}(0)

    ccall((:dgbsv_64_, liblapack), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ref{BlasInt}),
          BlasInt(n), BlasInt(kl), BlasInt(ku), BlasInt(1),
          AB, BlasInt(ldab), ipiv,
          x,  BlasInt(n), info)

    return x
end

# Float32
function gbsv_solve(A::BandedMatrix{Float32}, b::StridedVector{Float32})
    n = size(A, 1)
    kl, ku = bandwidths(A)
    ldab = 2*kl + ku + 1

    AB = zeros(Float32, ldab, n)
    bd = BandedMatrices.bandeddata(A)
    @inbounds @views AB[(kl+1):ldab, :] .= bd

    ipiv = Vector{BlasInt}(undef, n)
    x    = copy(b)
    info = Ref{BlasInt}(0)

    ccall((:sgbsv_64_, liblapack), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
           Ptr{Float32}, Ref{BlasInt}, Ptr{BlasInt},
           Ptr{Float32}, Ref{BlasInt}, Ref{BlasInt}),
          BlasInt(n), BlasInt(kl), BlasInt(ku), BlasInt(1),
          AB, BlasInt(ldab), ipiv,
          x,  BlasInt(n), info)

    return x
end

# n  = 4000
# kl = 10
# ku = 10

# A = brand(Float32, n, n, kl, ku) + 5.0I
# b = randn(n)

# x, info = gbsv_solve(A, b)
# # info == 0 means success
