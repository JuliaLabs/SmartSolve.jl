include("generate_default_code.jl")
using LinearAlgebra, SparseArrays
function ls_dev_prompt_maker(fn_str)
    return "You are a numerical linear algebra expert, and an expert Julia programmer." * 
            " The user will ask you to generate a function and use the following code the check if your solution is accurate and fast." * 
            " Here is the code: \n" * fn_str * "\nOnly return the function. Make sure the function name is proposed_fn. Do not return extra text." *
            # " Make sure that the function returns some statistics, so that calling proposed_fn(A, b) returns solution, stats." *
            # " Do not make the statistics a struct, just a dictionary" *
            " Assume that LinearAlgebra and SparseArrays is already imported."
end
function generate_linear_solver_code(prompt, checker_fn_str, secret_key, model = "gpt-5-mini"; max_iters = 3)
    return generate_default_code(prompt, checker_fn_str, secret_key, model, ls_dev_prompt_maker, max_iters)
end