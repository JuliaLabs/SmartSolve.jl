# using SmartSolve
# include("generate_default_code.jl")
# include("generate_linear_solver_code.jl")
# include("test_performance.jl")

prompt = "https://arxiv.org/abs/2504.08009 \n" * 
        "Using this paper, give me a high performance Julia implementation of the Ozaki Scheme II."

secret_key = ENV["OPENAI_API_KEY"]
checker_filename = "test_performance_mm.jl"

code, hist, timedout = generate_linear_solver_code(prompt, checker_filename, secret_key)