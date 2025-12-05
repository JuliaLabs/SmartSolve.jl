test_matrices = []
N = 2000#10_000
# push!(test_matrices, randn(N, N))
# push!(test_matrices, randn(N, N))
# push!(test_matrices, randn(N, N))
Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
  push!(test_matrices, randn(Blocks(N, N), N, N))
  push!(test_matrices, randn(Blocks(N, N), N, N))
  push!(test_matrices, randn(Blocks(N, N), N, N))
end

function evaluator(proposed_fn;
                        err_threshold::Float64 = 1.0,
                        runtime_threshold::Float64 = 1.1,
                        alloc_threshold::Float64 = 0.0)
    error_ratios  = Float64[]
    runtime_ratios = Float64[]
    alloc_ratios   = Float64[]
    for A_d in test_matrices
        # right-hand side on CPU
        A_dim2 = size(A_d, 2)
        b_d = Dagger.with_options(scope=Dagger.scope(;cuda_gpu=1)) do
            randn(Blocks(A_dim2), A_dim2)
        end
        # move to GPU; here we use a dense GPU matrix
        # If you have a sparse GPU solver, you can switch to CuSparseMatrixCSR(A_cpu)
        A_cuda = CuArray(collect(A_d))
        b_cuda = CuArray(collect(b_d))
        # --- Solve once to ensure kernels are compiled (warm-up) ---
        x_default = similar(b_cuda)
        CUSOLVER.gesv!(x_default, A_cuda, b_cuda, irs_precision = "R_32F")
        x_gen = similar(b_d)
        Base.invokelatest(proposed_fn, x_gen, A_d, b_d)
        CUDA.synchronize()
        # --- Error ratios (all on GPU, scalars on CPU) ---
        err_default = norm(A_cuda * x_default - b_cuda)
        err_gen     = norm(A_d * x_gen     - b_d)
        push!(error_ratios, err_default / err_gen)
        # --- Runtime ratios (GPU) ---
        b_default = @benchmark begin
            x = similar($b_cuda)
            CUSOLVER.gesv!($x, $A_cuda, $b_cuda, irs_precision = "R_32F")
            CUDA.synchronize()
        end
        b_gen = @benchmark begin
            x = similar($b_d)
            Base.invokelatest($proposed_fn, $x, $A_d, $b_d)
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