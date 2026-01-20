using LinearAlgebra
using SparseArrays
using BenchmarkTools
using OrderedCollections
using Plots

println("CPU benchmark with error-vs-time plot:\n")
include("dsgesv_solve.jl")
include("ex_solver.jl")

# Configuration
# N = 15_000
n_levels = [5000, 10000, 20000]
# umfpack_control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int64) # read Julia default configuration for a Float64 sparse matrix
# #SparseArrays.UMFPACK.show_umf_ctrl(umfpack_control) # optional - display values
# umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP] = 2.0 # reenable iterative refinement (2 is UMFPACK default max iterative refinement steps)

solvers = OrderedDict(
    # "UMFPACK (Baseline Direct Solve)" => (A, b) -> (A \ b),
    # "UMFPACK + Iterative Refinement" => (A, b) -> lu(A; control = umfpack_control) \ b,
    "LUIR" => (A, b) -> proposed_fn(A, b),
    # "LUIR, 20 iters" => (A, b) -> proposed_fn(A, b, iters = 20),
    # "LUIR, 100 iters" => (A, b) -> proposed_fn(A, b, iters = 100),
    "dsgesv" => (A, b) -> dsgesv_solve(A, b),
    "baseline" => (A, b) -> lu(A) \ b
)

# Store results for plotting
results = Dict()

for N in n_levels
    println("\n=== Matrix size: $N ===")
    
    # Generate problem
    A = randn(N, N)
    b = randn(N)
    
    results[N] = Dict()
    
    for (solver_name, solver_fn) in solvers
        println("  $solver_name...")
        
        if solver_name == "dsgesv"
            A = Matrix(A)
        end
        # Warm-up
        b_warm = copy(b)
        try
            x_warm = solver_fn(A, b_warm)
        catch e
            println("    Warning: solver failed during warm-up: $e")
            continue
        end
        
        # Benchmark
        b_bench = copy(b)
        try
            bench = @benchmark begin
                x = $(solver_fn)($A, $b_bench)
            end
            
            time_ms = median(bench.times) / 1e9  # Convert to s
            
            # Compute error
            b_err = copy(b)
            x_sol = solver_fn(A, b_err)
            error = norm(A*x_sol - b_err) / norm(b_err)
            
            results[N][solver_name] = (time=time_ms, error=error)
            println("    Time: $(round(time_ms, digits=3)) s, Error: $(round(error, sigdigits=3))")
        catch e
            println("    Error during benchmark: $e")
        end
    end

    GC.gc()
end

# Create error-vs-time plot
p = plot(
    size=(1600, 1200),
    #legend=:topright,
    legend=:bottomright,
    xlabel="Time (s)",
    ylabel="Relative residual: ||Ax - b||₂ / ||b||₂",
 #   xscale=:log10,
    yscale=:log10,
    guidefontsize=22,#18,
    tickfontsize=20, #16,
    legendfontsize=14, #14,
    margin=5*Plots.mm,
    framestyle=:box,
    title="Dense Random Matrices of Varying Size (N) and\n CPU Solvers",
    titlefontsize=22,
)

# Symbols encode sparsity levels; colors encode solvers.
# Define marker for each sparsity and a color for each solver.
## Sparsity shapes
marker_map_sparsity = OrderedDict(n_levels[1]=>:circle, n_levels[2]=>:square, n_levels[3]=>:utriangle)
## Solver color shades
colors = [:red, :blue, :green, :purple, :orange, :black]

# Plot each point individually so marker shape shows sparsity and color shows solver.
for (i, solver_name) in enumerate(keys(solvers))
    for N in n_levels
        if N in keys(results) && solver_name in keys(results[N])
            t = results[N][solver_name].time
            e = results[N][solver_name].error
            scatter!(p, [t], [e];
                     label="",
                     marker=marker_map_sparsity[N],
                     markersize=15,
                     color=colors[i],
                     markerstrokecolor=:black,
                     markerstrokewidth=0.0,#0.8,
                     alpha=0.45)
        end
    end
end

# Create a combined legend
for (i, solver_name) in enumerate(keys(solvers))
    for N in n_levels
        lbl = "$(solver_name), N:$(N)"
        scatter!(p, [NaN], [NaN]; label=lbl,
                 marker=marker_map_sparsity[N],
                 markersize=15,
                 color=colors[i],
                 markerstrokecolor=:black,
                 markerstrokewidth=0.0, 
                 alpha=0.45)
    end
end

savefig(p, "error_vs_time_dense.pdf")
println("\n✓ Plot saved as error_vs_time_dense_8_threads.pdf")

display(p)