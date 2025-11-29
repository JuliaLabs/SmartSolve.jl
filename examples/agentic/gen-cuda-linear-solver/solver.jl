function proposed_fn(A_d, b_d)
    @assert CUDA.has_cuda() "CUDA not available"

    # Ensure A and b are dense GPU arrays (CuArray). Convert any CPU/sparse inputs.
    A_gpu = isa(A_d, CuArray) && ndims(A_d) == 2 ? A_d : cu(Matrix(A_d))
    b_gpu = isa(b_d, CuArray) ? b_d : cu(b_d)

    n = size(A_gpu, 2)
    @assert length(b_gpu) == n

    T = eltype(A_gpu)

    if T === Float64
        # Mixed precision: factorize in Float32 for speed, do residuals in Float64 for accuracy.
        As = Float32.(A_gpu)           # single-precision matrix on GPU
        bs = Float32.(b_gpu)           # single-precision rhs on GPU

        F = lu(As)                     # single-precision LU (GPU)
        CUDA.synchronize()

        # initial solution in single precision, then promote
        x_s = F \ bs                   # CuArray{Float32}
        x = Float64.(x_s)              # CuArray{Float64}

        # preallocate temporaries on GPU
        tmp = similar(b_gpu)           # Float64
        r   = similar(b_gpu)           # Float64
        r_s = similar(bs)              # Float32

        for i in 1:5
            # tmp = A * x  (in-place)
            mul!(tmp, A_gpu, x)
            @. r = b_gpu - tmp          # residual in Float64
            @. r_s = Float32(r)         # convert residual to Float32
            delta_s = F \ r_s           # solve in single precision (CuArray{Float32})
            @. x += Float64.(delta_s)   # promote and update solution
        end

        CUDA.synchronize()
        return x

    elseif T === Float32
        # Pure single-precision factorization and refinement
        F = lu(A_gpu)
        CUDA.synchronize()

        x = F \ b_gpu

        tmp = similar(b_gpu)
        r   = similar(b_gpu)

        for i in 1:5
            mul!(tmp, A_gpu, x)
            @. r = b_gpu - tmp
            delta = F \ r
            @. x += delta
        end

        CUDA.synchronize()
        return x

    else
        # Fallback: operate in Float64 on GPU
        Ad = Float64.(A_gpu)
        bd = Float64.(b_gpu)

        F = lu(Ad)
        CUDA.synchronize()

        x = F \ bd
        tmp = similar(bd)
        r   = similar(bd)

        for i in 1:10
            mul!(tmp, Ad, x)
            @. r = bd - tmp
            delta = F \ r
            @. x += delta
        end

        CUDA.synchronize()
        return x
    end
end