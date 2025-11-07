using LinearAlgebra, BenchmarkTools, MatrixDepot, CUDA, SparseArrays

function benchmark_ms( myfunc, args...;kwargs...)
    elapsed=0.0
    best=100000
    i=0
    numruns = 20
    while(elapsed<200.0 || i<2)
        CUDA.synchronize()
        start = time_ns()
        for i=1:numruns
            myfunc(args...;kwargs...)
            CUDA.synchronize()
        end
        endtime = time_ns()
        thisduration=(endtime-start)/1e6
        elapsed += thisduration
        best = min(thisduration/numruns,best)
        i+=1
    end
    return best
end

test_matrix_names = ["Bai/af23560", "Engwirda/airfoil_2d", "vanHeukelum/cage10"]
test_matrices = matrixdepot.(test_matrix_names)
push!(test_matrices, sprand(20000, 20000, 0.1))
push!(test_matrices, sprand(20000, 20000, 0.1))
# push!(test_matrices, sprand(10000, 10000, 0.1))

cuda_test_matrices = CuArray.(test_matrices)


base_prompt(rel_errs, speedups) = "Here are the errors compared to built-in linear solver: 
$(rel_errs)
and here are the speed-up ratio compared to built-in solver:
$(speedups)."
function evaluator(proposed_fn, tol = 1e-6)
    rel_errors = Float64[]
    speedups = Float64[]
    for A in cuda_test_matrices
        b = CUDA.randn(Float64, size(A,2))
        
        x_exact = A \ b
        alg_solver(A, b) = Base.invokelatest(proposed_fn, A, b)

        x_alg = alg_solver(A, b)
        

        push!(rel_errors, norm(x_alg - x_exact)/norm(x_exact))

        benchmark_ms(\, A, b)
        base_runtime = benchmark_ms(\, A, b)


        benchmark_ms(alg_solver, A, b)


        alg_runtime = benchmark_ms(alg_solver, A, b)
        push!(speedups, base_runtime/alg_runtime)


        # println("done")
    end
    return mean(rel_errors) < tol && mean(speedups) > 1.1, base_prompt(rel_errors, speedups)
end 