using LinearAlgebra

const BlasInt = LinearAlgebra.BlasInt
const liblapack = LinearAlgebra.LAPACK.liblapack

function dsgesv_solve(A::StridedMatrix{Float64}, b::StridedVector{Float64})
    n    = size(A, 1)
    nrhs = BlasInt(1)

    ipiv  = Vector{BlasInt}(undef, n)
    x     = Vector{Float64}(undef, n)
    work  = Matrix{Float64}(undef, n, 1)
    swork = Vector{Float32}(undef, n * (n + 1))
    iter  = Ref{BlasInt}(0)
    info  = Ref{BlasInt}(0)

    lda = BlasInt(max(1, n))
    ldb = BlasInt(max(1, n))
    ldx = BlasInt(max(1, n))

    ccall((:dsgesv_64_, liblapack), Cvoid,
          (Ref{BlasInt}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt},
           Ptr{BlasInt},
           Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ptr{Float32},
           Ref{BlasInt}, Ref{BlasInt}),
          BlasInt(n), nrhs,
          A, lda,
          ipiv,
          b, ldb,
          x, ldx,
          work, swork,
          iter, info)

    return x
end

# # -----------------------
# # Example usage
# # -----------------------
# n = 200
# A = randn(n, n)
# b = randn(n)

# x, iter, info = dsgesv_solve(A, b)

# println("info = ", info)  # 0 means success
# println("iter = ", iter)  # >0 refinement iters, <0 fallback to full double
# println("relres = ", norm(A*x - b)/norm(b))