function proposed_fn(A, b)
    # Cache Float32 factorizations and work buffers per matrix identity
    if !isdefined(@__MODULE__, :LU32_CACHE)
        global LU32_CACHE = IdDict{UInt64, Tuple{Any, Vector{Float64}, Vector{Float32}, Vector{Float32}}}()
    end

    # Ensure b as Float64 vector (avoid copy if already Float64)
    b64 = eltype(b) === Float64 ? b : Vector{Float64}(b)
    n = length(b64)

    key = objectid(A)
    F32, r64, work32, bf32 = get!(LU32_CACHE, key) do
        # Build a single-precision copy of the numeric values (structure reuse)
        nz32 = Float32.(A.nzval)
        Af = SparseMatrixCSC{Float32, Int}(size(A,1), size(A,2),
                                           copy(A.colptr), copy(A.rowval),
                                           nz32)
        F32_local = lu(Af)                           # single-precision sparse LU
        r64_local = Vector{Float64}(undef, n)       # residual buffer (double)
        work32_local = Vector{Float32}(undef, n)    # temp residual in single
        bf32_local = Vector{Float32}(undef, n)      # temp right-hand side in single
        return (F32_local, r64_local, work32_local, bf32_local)
    end

    # Initial solve in single precision, accumulate in double
    @inbounds for i = 1:n
        bf32[i] = Float32(b64[i])
    end
    xf32 = F32 \ bf32
    x = Float64.(xf32)

    # Iterative refinement: compute residual in double, solve correction in single, update double solution
    for _ = 1:5
        mul!(r64, A, x)                       # r64 = A * x (double)
        @inbounds for i = 1:n
            r64[i] = b64[i] - r64[i]          # r64 = b - A*x
            work32[i] = Float32(r64[i])       # convert residual to single
        end
        d32 = F32 \ work32
        @inbounds for i = 1:n
            x[i] += Float64(d32[i])           # update solution in double
        end
    end

    return x
end