# using SmartSolve
include("generate_default_code.jl")
# include("test_performance.jl")

prompt = "Implement a merge sort algorithm in Julia."

secret_key = ENV["OPENAI_API_KEY"]
checker_filename = "test_sorted.jl"

code, hist, timedout = generate_default_code(prompt, checker_filename, secret_key)