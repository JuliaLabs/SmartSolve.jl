using SmartSolve

prompt = "https://nhigham.com/2023/03/13/what-is-iterative-refinement/ \n" * 
        "Using this blog, give me a high performance Julia implementation of LU + iterative refinement."

secret_key = ENV["OPENAI_API_KEY"]
# checker_filename = "test_performance.jl"

code, hist, timedout = generate_linear_solver_code(prompt, secret_key)