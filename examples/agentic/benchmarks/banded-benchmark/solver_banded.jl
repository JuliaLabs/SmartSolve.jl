include("LinearSolveSLUIRExt_banded.jl")

function proposed_fn(A, b; iters = 5)
    prob = LinearProblem(A, b)
    sol = solve(prob, SLUIRFactorization(iterations = iters))
    return sol.u
end