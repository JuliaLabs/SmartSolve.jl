using LinearAlgebra, BenchmarkTools, MatrixMarket
proj_dir = @__DIR__ 
matrix_dir = proj_dir * "/Matrices/"
test_matrix_names = matrix_dir .* ["af23560.mtx", "airfoil_2d.mtx", "cage10.mtx"]
test_matrices = mmread.(test_matrix_names)
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
        x_alg = proposed_fn(A, b)
        push!(rel_errors, norm(x_alg - x_exact)/norm(x_exact))
        base_runtime = @btimed $A \ $b
        alg_runtime = @btimed $proposed_fn($A, $b)
        push!(speedups, base_runtime.time/alg_runtime.time)
        println("done")
    end
    return mean(rel_errors) < tol, base_prompt(rel_errors, speedups)
end