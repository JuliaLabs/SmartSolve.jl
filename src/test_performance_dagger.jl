function evaluator( proposed_fn;
                    err_threshold::Float64 = 1.0,
                    runtime_threshold::Float64 = 1.1,
                    alloc_threshold::Float64 = 0.0)
    N = 2000 #10_000
    error_ratios   = Float64[]
    runtime_ratios = Float64[]
    alloc_ratios   = Float64[]
    for _ in 1:3

        # SPD Matrix
        A = randn(N, N)
        A = A*A' + N*I

        # Right-hand side
        b = randn(N)

        # SPD Matrix and right hand side on GPU (CUDA)
        A_cuda = CuArray(A)
        b_cuda = CuArray(b)

        # SPD Matrix and right-hand side on GPU (Dagger Distributed)
        A_d = Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
            distribute(A, Blocks(N÷4, N÷4))
        end
        b_d = Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
            distribute(b, Blocks(N÷4))
        end
        
        # --- Solve once to ensure kernels are compiled (warm-up) ---
        x_default = cholesky(A_cuda) \ b_cuda
        x_gen = Base.invokelatest(proposed_fn, A_d, b_d)
        CUDA.synchronize()

        # --- Error ratios (all on GPU, scalars on CPU) ---
        err_default = norm(A_cuda * x_default - b_cuda)
        err_gen     = norm(A_d * x_gen - b_d)
        push!(error_ratios, err_default / err_gen)
        # --- Runtime ratios (GPU) ---
        b_default = @benchmark begin
            cholesky($A_cuda) \ $b_cuda
            CUDA.synchronize()
        end
        b_gen = @benchmark begin
            Base.invokelatest($proposed_fn, $A_d, $b_d)
        end
        push!(runtime_ratios, median(b_default.times) / median(b_gen.times))
        push!(alloc_ratios,   median(b_default.allocs) / median(b_gen.allocs))
    end
    m_err     = median(error_ratios)
    m_runtime = median(runtime_ratios)
    m_alloc   = median(alloc_ratios)
    report = get_report(m_err, m_runtime, m_alloc,
                        err_threshold, runtime_threshold, alloc_threshold)
    println(report)
    ok = (m_err     >= err_threshold)      &&  # 1.0 => no worse error
         (m_runtime >= runtime_threshold)  &&  # 1.1 => at least 10% faster
         (m_alloc   >= alloc_threshold)        # 0.0 => no alloc requirement
    return ok, report
end