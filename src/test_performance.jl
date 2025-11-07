test_matrix_names = ["Bai/af23560", "Engwirda/airfoil_2d", "vanHeukelum/cage10"]
test_matrices = matrixdepot.(test_matrix_names)
push!(test_matrices, sprand(1000, 1000, 0.1))
push!(test_matrices, sprand(1000, 1000, 0.1))
# push!(test_matrices, sprand(10000, 10000, 0.1))

base_prompt(rel_errs, speedups) = "Here are the errors compared to built-in linear solver: 
$(rel_errs)
and here are the speed-up ratio compared to built-in solver:
$(speedups)."
function evaluator(proposed_fn, tol = 1e-6)
    rel_errors = Float64[]
    speedups = Float64[]
    for A in test_matrices
        b = randn(size(A,2))
        x_exact = A \ b
        x_alg = Base.invokelatest(proposed_fn, A, b)
        # x_alg, _ = proposed_fn(A, b)
        push!(rel_errors, norm(x_alg - x_exact)/norm(x_exact))
        base_runtime = @btimed $A \ $b
        alg_runtime = @btimed $Base.invokelatest($proposed_fn, $A, $b)
        push!(speedups, base_runtime.time/alg_runtime.time)
        # println("done")
    end
    return mean(rel_errors) < tol && mean(speedups) > 1.1, base_prompt(rel_errors, speedups)
end 