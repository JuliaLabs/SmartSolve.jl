function dev_prompt_maker(fn_string) 
    return "You are an expert software developer." * 
            " The user will ask you to generate a function and use the following code the check if your solution is correct." * 
            " Note that the function can also return some performace description." *
            " Here is the code: \n" * fn_string * "\nOnly return the function." *
            " Call the function proposed_fn." *
            " Do not use anonymous function definitions. Do not return extra text."
end

function description_prompt_maker(check, description)
    if check
        return "Passed." * " Here is the description the checker function returned:\n" * description
    end
    return "I ran the code using a checker function. It failed. I need you to fix it." *
            " Here is the description the checker function returned:\n" * description
end

function error_prompt_maker(err_message)
    return "Your code gave the following error: " * err_message
end

proposed_fn(x) = x
evaluator(x) = (true, "")
function generate_default_code(prompt, secret_key, checker_filename;
                              model = "gpt-5-mini", dev_prompt_fn = dev_prompt_maker, max_iters = 3)
    """
        - checker_fn: proposed_fn -> check : Bool, performance_description : String
    """
    include(checker_filename)
    checker_fn_str = read(checker_filename, String)
    dev_prompt = dev_prompt_fn(checker_fn_str)
    chat_history = [Dict("role" => "developer", "content"=> dev_prompt),
                    Dict("role" => "user", "content"=> prompt),]
    
    converged = false
    for iters = 1:max_iters
        # generate code and evaluate
        r = create_chat(secret_key,
                        model,
                        chat_history)
        gen_code = r.response[:choices][begin][:message][:content] 
        
        println("Iteration $iters")

        # println(gen_code)
        push!(chat_history, Dict("role" => "assistant", "content" => gen_code)) 
        try
            eval(Meta.parse(gen_code))
            # # check how the code did, if good enough, break
            
            check, performance_description = invokelatest(evaluator, proposed_fn)
            check && println("check passed")

            next_prompt = description_prompt_maker(check, performance_description)
            push!(chat_history, Dict("role" => "user", "content" => next_prompt))
            converged = ~(iters == max_iters)
            check && break
        catch e
            error_msg = sprint(showerror, e)
            st = sprint((io,v) -> show(io, "text/plain", v), stacktrace(catch_backtrace()))
           
            next_prompt = error_prompt_maker(error_msg * "\n" * st)
            push!(chat_history, Dict("role" => "user", "content" => next_prompt))
        end
        converged = ~(iters == max_iters)
    end
    final_code = chat_history[end - 1]["content"]
    return final_code, chat_history, converged
end

function ls_dev_prompt_maker(fn_str)
    return "You are a numerical linear algebra expert, and an expert Julia programmer." * 
            " The user will ask you to generate a function and use the following code the check if your solution is accurate and fast." * 
            " Here is the code: \n" * fn_str * "\nOnly return the function. Make sure the function name is proposed_fn. Do not return extra text." *
            " Assume that LinearAlgebra and SparseArrays is already imported."
end

function ls_cuda_dev_prompt_maker(fn_str)
    return "You are a numerical linear algebra expert, and an expert Julia programmer. You are very experienced in GPU programming using CUDA." * 
            " The user will ask you to generate a function and use the following code the check if your solution is accurate and fast." * 
            " Make sure the code you produce uses CUDA." *
            " Here is the code: \n" * fn_str * "\nOnly return the function. Make sure the function name is proposed_fn. Do not return extra text." *
            " Assume that LinearAlgebra and SparseArrays is already imported."
end

function ls_dagger_dev_prompt_maker(fn_str)
    return "You are a numerical linear algebra expert, and an expert Julia programmer. You are very experienced in GPU programming using CUDA." * 
            " The user will ask you to generate a function and use the following code the check if your solution is accurate and fast." * 
            " Make sure the code you produce uses Dagger." *
            " Here is the code: \n" * fn_str * "\nOnly return the function. Make sure the function name is proposed_fn. Do not return extra text." *
            " Assume that LinearAlgebra and SparseArrays is already imported." *
            " Assume that Dagger is already imported." *
            " Use the following Dagger.jl documentation: https://juliaparallel.org/Dagger.jl/dev/" *
            " Use the following Dagger.jl implementation of Cholesky as an example: https://github.com/JuliaParallel/Dagger.jl/blob/67211816781d59109d74940550ca2d80af96b13d/src/array/cholesky.jl"
end

src_dir = @__DIR__

function gen_linear_solver(prompt, secret_key;
                          checker_filename = src_dir * "/test_performance.jl",
                          model = "gpt-5-mini",
                          max_iters = 10)
    return generate_default_code(prompt, secret_key, checker_filename;
                                model = model,
                                dev_prompt_fn=ls_dev_prompt_maker,
                                max_iters = max_iters)
end

function gen_linear_solver_cuda(prompt, secret_key;
                                checker_filename = src_dir *"/test_performance_cuda.jl",
                                model = "gpt-5-mini",
                                max_iters = 10)
    return generate_default_code(prompt, secret_key, checker_filename;
                                 model=model,
                                 dev_prompt_fn=ls_cuda_dev_prompt_maker,
                                 max_iters = max_iters)
end

function gen_linear_solver_dagger(prompt, secret_key;
                                  checker_filename = src_dir *"/test_performance_dagger.jl",
                                  model = "gpt-5-mini",
                                  max_iters = 10)
    return generate_default_code(prompt, secret_key, checker_filename;
                                 model=model,
                                 dev_prompt_fn=ls_dagger_dev_prompt_maker,
                                 max_iters = max_iters)
end

function printhist(hist)
    for (i, (role, message)) in enumerate(hist)
            println("Message $i $(role[2]):\n$(message[2])\n")
    end
end

function get_report(m_err, m_runtime, m_alloc,
                    err_threshold, runtime_threshold, alloc_threshold)
    report = """
    Median error ratio (error_default / error_gen): $(m_err)
    Desired median error ratio: >= $err_threshold
    Median runtime ratio or speedup (runtime_default / runtime_gen): $(m_runtime)
    Desired median runtime ratio: >= $runtime_threshold
    Allocation median ratio (alloc_default / alloc_gen): $(m_alloc)
    Desired median allocation ratio: >= $alloc_threshold
    """
    return report
end