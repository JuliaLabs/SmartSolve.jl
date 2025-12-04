function proposed_fn(A::SparseMatrixCSC, b::AbstractVector)
    @assert size(A,1) == size(A,2) "A must be square"
    n = length(b)
    @assert size(A,2) == n "Dimensions of A and b must agree"

    niters = 4

    # Convert sparse matrix to dense double for accurate residual computation
    # and to dense single for fast factorization/solves with multithreaded BLAS.
    Ad64 = Array(A)                    # dense Float64
    Ad32 = Array{Float32}(undef, n, n)
    @inbounds for j in 1:n
        for i in 1:n
            Ad32[i,j] = Float32(Ad64[i,j])
        end
    end

    # Convert rhs to Float32 once
    b32 = Vector{Float32}(undef, n)
    @inbounds @simd for i in 1:n
        b32[i] = Float32(b[i])
    end

    # Factorize dense single-precision matrix (uses LAPACK/BLAS and is multithreaded)
    F32 = lu(Ad32)

    # Initial solve in single precision, in-place if possible
    x32 = copy(b32)
    try
        LinearAlgebra.ldiv!(F32, x32)   # in-place: x32 <- Ad32 \ b32
    catch
        x32 = F32 \ b32                 # fallback
    end

    # Promote to double precision for accumulation and residual computation
    x = Vector{Float64}(undef, n)
    @inbounds @simd for i in 1:n
        x[i] = Float64(x32[i])
    end

    # Preallocate working vectors
    r = similar(b)                     # Float64 residual
    r32 = Vector{Float32}(undef, n)    # single-precision correction (in-place)

    for iter in 1:niters
        # r = b - Ad64 * x   (use BLAS for dense matvec)
        mul!(r, Ad64, x)              # r = Ad64 * x
        @inbounds @simd for i in 1:n
            r[i] = b[i] - r[i]
            r32[i] = Float32(r[i])
        end

        # Solve correction in single precision using the LU factorization
        try
            LinearAlgebra.ldiv!(F32, r32)   # r32 <- Ad32 \ r32 (in-place)
        catch
            r32 = F32 \ r32                 # fallback
        end

        # Update double-precision solution
        @inbounds @simd for i in 1:n
            x[i] += Float64(r32[i])
        end
    end

    return x
end