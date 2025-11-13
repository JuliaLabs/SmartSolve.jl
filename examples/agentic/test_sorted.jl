function test_sorted(a)
    return sum(1 .- (a .== Base.sort(a))) == 0
end
function evaluator(proposed_fn)
    n = 1000
    a = rand(1:n, n)
    a = Base.invokelatest(proposed_fn, a)
    # a = sorting_fn(a)
    return test_sorted(a), ""
end