# using SmartSolve
include("generate_default_code.jl")
include("test_performance.jl")

prompt = "Implement a merge sort algorithm in Julia."

secret_key = ENV["OPENAI_API_KEY"]

generate_code(prompt, checker, secret_key)


###

# smart_solve(prompt)
# "1) identify_problem(prompt) -> :factorization, :sparse_linear_solver, :dense_solver "
# "2) specialized_code_generator(prompt) -> produce specific prompt + checker"
# "3) gen_code(outputs of step 2)"