test_matrices = []
N = 10_000
push!(test_matrices, sprand(N, N, 0.1))
push!(test_matrices, sprand(N, N, 0.2))
push!(test_matrices, sprand(N, N, 0.3))

function get_report(m_err, m_runtime, m_alloc,
                    err_threshold, runtime_threshold, alloc_threshold)
    report = """
    Median error ratio (error_default / error_gen): $(m_err)
    Desired median error ratio: >= $err_threshold
    Median runtime ratio or speedup (runtime_default / runtime_gen): $(m_runtime)
    Desired median runtime ratio: >= $runtime_threshold
    Allocation median ratio (alloc_default / alloc_gen): $(m_alloc)
    Desired median allocation ratio: >= $alloc_threshold
    """
    return report
end

function evaluator_cuda(proposed_fn;
                        err_threshold::Float64 = 1.0,
                        runtime_threshold::Float64 = 1.1,
                        alloc_threshold::Float64 = 0.0)

    error_ratios  = Float64[]
    runtime_ratios = Float64[]
    alloc_ratios   = Float64[]

    for A_cpu in test_matrices
        # right-hand side on CPU
        b_cpu = randn(size(A_cpu, 2))

        # move to GPU; here we use a dense GPU matrix
        # If you have a sparse GPU solver, you can switch to CuSparseMatrixCSR(A_cpu)
        A_d = cu(Matrix(A_cpu))
        b_d = cu(b_cpu)

        # --- Solve once to ensure kernels are compiled (warm-up) ---
        x_default = A_d \ b_d
        x_gen     = Base.invokelatest(proposed_fn, A_d, b_d)
        CUDA.synchronize()

        # --- Error ratios (all on GPU, scalars on CPU) ---
        err_default = norm(A_d * x_default - b_d)
        err_gen     = norm(A_d * x_gen     - b_d)
        push!(error_ratios, err_default / err_gen)

        # --- Runtime ratios (GPU) ---
        b_default = @benchmark begin
            x = $A_d \ $b_d
            CUDA.synchronize()
        end

        b_gen = @benchmark begin
            x = Base.invokelatest($proposed_fn, $A_d, $b_d)
            CUDA.synchronize()
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
