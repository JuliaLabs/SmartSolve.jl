include("LinearSolveSLUIRExt_mumps.jl")

function proposed_fn(A, b; iters = 5)
    prob = LinearProblem(A, b)
    sol = SciMLBase.solve(prob, SLUIRFactorization(iterations = iters))
    return sol.u
end