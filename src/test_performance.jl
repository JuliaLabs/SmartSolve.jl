# test_matrix_names = ["Bai/af23560", "Engwirda/airfoil_2d", "vanHeukelum/cage10"]
# test_matrices = matrixdepot.(test_matrix_names)
test_matrices = []
push!(test_matrices, sprand(10000, 10000, 0.1))
push!(test_matrices, sprand(10000, 10000, 0.1))
push!(test_matrices, sprand(10000, 10000, 0.1))

base_prompt(rel_errs, speedups, alloc_ratios) = "Here are the errors compared to built-in linear solver: 
$(rel_errs)
and here are the speed-up ratio compared to built-in solver:
$(speedups).
the ratio of allocations (base_alloc/proposed_alloc) is:
$(alloc_ratios)."

function evaluator(proposed_fn, tol = 1e-6)
    rel_errors = Float64[]
    speedups = Float64[]
    alloc_ratios = Float64[]
    for A in test_matrices
        b = randn(size(A,2))
        x_exact = A \ b
        x_alg = Base.invokelatest(proposed_fn, A, b)

        push!(rel_errors, norm(x_alg - x_exact)/norm(x_exact))
        base_runtime = @btimed $A \ $b
        alg_runtime = @btimed $Base.invokelatest($proposed_fn, $A, $b)
        push!(speedups, base_runtime.time/alg_runtime.time)

        base_alloc = @ballocated $A \ $b
        alg_alloc = @ballocated $Base.invokelatest($proposed_fn, $A, $b)
        push!(alloc_ratios, base_alloc/ alg_alloc)
        # println("done")
    end
    return mean(rel_errors) < tol && mean(speedups) > 1.1, base_prompt(rel_errors, speedups, alloc_ratios)
end 