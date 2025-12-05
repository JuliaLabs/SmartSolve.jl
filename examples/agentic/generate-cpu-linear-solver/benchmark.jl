using LinearAlgebra
using SparseArrays
using CUDA
using BenchmarkTools
using OrderedCollections
using Plots

println("CPU benchmark with error-vs-time plot:\n")

include("solver.jl")

# Configuration
N = 15_000
sparsity_levels = [0.1, 0.5, 0.9]

umfpack_control = SparseArrays.UMFPACK.get_umfpack_control(Float64, Int64) # read Julia default configuration for a Float64 sparse matrix
#SparseArrays.UMFPACK.show_umf_ctrl(umfpack_control) # optional - display values
umfpack_control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP] = 2.0 # reenable iterative refinement (2 is UMFPACK default max iterative refinement steps)

solvers = OrderedDict(
    "Default" => (A, b) -> (A \ b),
    "UMFPACK" => (A, b) -> lu(A; control = umfpack_control) \ b,
    "Generated" => (A, b) -> proposed_fn(A, b)
)

# Store results for plotting
results = Dict()

for sparsity in sparsity_levels
    println("\n=== Sparsity: $sparsity ===")
    
    # Generate problem
    A = sprand(N, N, sparsity)
    b = rand(N)
    
    results[sparsity] = Dict()
    
    for (solver_name, solver_fn) in solvers
        println("  $solver_name...")
        
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
            end seconds = 5 samples = 10
            
            time_ms = median(bench.times) / 1e9  # Convert to s
            
            # Compute error
            b_err = copy(b)
            x_sol = solver_fn(A, b_err)
            error = norm(A*x_sol - b_err) / norm(b_err)
            
            results[sparsity][solver_name] = (time=time_ms, error=error)
            println("    Time: $(round(time_ms, digits=3)) s, Error: $(round(error, sigdigits=3))")
        catch e
            println("    Error during benchmark: $e")
        end
    end
end

# Create error-vs-time plot
p = plot(
    size=(800, 800),
    #legend=:topright,
    legend=:bottomleft,
    xlabel="Time (s)",
    ylabel="Relative residual: ||Ax - b||₂ / ||b||₂",
 #   xscale=:log10,
    yscale=:log10,
    guidefontsize=22,#18,
    tickfontsize=20, #16,
    legendfontsize=18, #14,
    margin=5*Plots.mm,
    framestyle=:box,
    title="Random Matrices of Size $(N)x$(N),\n Varying Sparsity Levels (ρ) and\n CPU Solvers",
    titlefontsize=22,
)

# Symbols encode sparsity levels; colors encode solvers.
# Define marker for each sparsity and a color for each solver.
## Sparsity shapes
marker_map_sparsity = OrderedDict(0.1=>:circle, 0.5=>:square, 0.9=>:utriangle)
## Solver color shades
color_map_solver = OrderedDict("Default"=>:red, "UMFPACK"=>:blue, "Generated"=>:green)

# Plot each point individually so marker shape shows sparsity and color shows solver.
for solver_name in keys(solvers)
    for sparsity in sparsity_levels
        if sparsity in keys(results) && solver_name in keys(results[sparsity])
            t = results[sparsity][solver_name].time
            e = results[sparsity][solver_name].error
            scatter!(p, [t], [e];
                     label="",
                     marker=marker_map_sparsity[sparsity],
                     markersize=15,
                     color=color_map_solver[solver_name],
                     markerstrokecolor=:black,
                     markerstrokewidth=0.0,#0.8,
                     alpha=0.45)
        end
    end
end

# Create a combined legend
for solver_name in keys(solvers)
    for s in sparsity_levels
        lbl = "$(solver_name), ρ:$(s)"
        scatter!(p, [NaN], [NaN]; label=lbl,
                 marker=marker_map_sparsity[s],
                 markersize=15,
                 color=color_map_solver[solver_name],
                 markerstrokecolor=:black,
                 markerstrokewidth=0.0, 
                 alpha=0.45)
    end
end

savefig(p, "error_vs_time.pdf")
println("\n✓ Plot saved as error_vs_time.pdf")

display(p)