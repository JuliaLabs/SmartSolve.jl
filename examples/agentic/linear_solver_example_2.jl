# using SmartSolve
include("generate_default_code.jl")
include("generate_linear_solver_code.jl")
# include("test_performance.jl")

prompt = "https://nhigham.com/2023/03/13/what-is-iterative-refinement/ \n" * 
        "Using this blog, give me a high performance Julia implementation of LU + iterative refinement."

secret_key = ENV["OPENAI_API_KEY"]
checker_filename = "test_performance.jl"

code, hist, timedout = generate_linear_solver_code(prompt, checker_filename, secret_key)


###

# smart_solve(prompt)
# "1) identify_problem(prompt) -> :factorization, :sparse_linear_solver, :dense_solver "
# "2) specialized_code_generator(prompt) -> produce specific prompt + checker"
# "3) gen_code(outputs of step 2)"