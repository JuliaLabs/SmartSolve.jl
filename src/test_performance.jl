test_matrices = []
N = 5000
push!(test_matrices, sprand(N, N, 0.1))
push!(test_matrices, sprand(N, N, 0.2))
push!(test_matrices, sprand(N, N, 0.3))

function get_report(m_err, m_runtime, m_alloc,
                    err_threshold, runtime_threshold, alloc_threshold)
    report = """
    Median error ratio (error_default / error_gen): $(m_err)
    Desired median error ratio: >= $err_threshold
    Median Runtime ratio or speedup (runtime_default / runtime_gen): $(m_runtime)
    Desired median runtime ratio: >= $runtime_threshold
    Allocation median ratio (alloc_default / alloc_gen): $(m_alloc)
    Desired median allocation ratio: >= $alloc_threshold
    """
    return report
end

function evaluator(proposed_fn, err_threshold=1.0,
                                runtime_threshold=1.1,
                                alloc_threshold=0.0)
    error_ratios = Float64[]
    runtime_ratios = Float64[]
    alloc_ratios = Float64[]
    for A in test_matrices
        b = randn(size(A,2))
        x_default = A \ b
        x_gen = Base.invokelatest(proposed_fn, A, b)

        err_default = norm(A * x_default - b)
        err_gen = norm(A * x_gen - b)
        push!(error_ratios, err_default/err_gen)
        
        b_default = @benchmark $A \ $b
        b_gen = @benchmark $Base.invokelatest($proposed_fn, $A, $b)
        push!(runtime_ratios, median(b_default.times)/median(b_gen.times))
        push!(alloc_ratios, median(b_default.allocs)/median(b_gen.allocs))
    end
    m_err = median(error_ratios)
    m_runtime = median(runtime_ratios)
    m_alloc = median(alloc_ratios)
    report = get_report(m_err, m_runtime, m_alloc, err_threshold,
                        runtime_threshold, alloc_threshold)
    println(report)
    return m_err     >= err_threshold &&     # 1.0 means no worse error
           m_runtime >= runtime_threshold && # 1.1 means at least 10% faster
           m_alloc   >= m_alloc,             # 0.0 means no allocation requirement
           report
end