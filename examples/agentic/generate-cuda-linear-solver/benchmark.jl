using LinearAlgebra
using SparseArrays
using CUDA
using BenchmarkTools
using OrderedCollections
using Plots

println("GPU benchmark with error-vs-time plot:\n")

include("solver.jl")

# Configuration
N = 15_000
sparsity_levels = [0.1, 0.5, 0.9]
solvers = OrderedDict(
    "Default" => (Ad, bd) -> (Ad \ bd),
    "gesv!" => (Ad, bd) -> begin
        x = CuArray(zeros(size(Ad, 1)))
        CUDA.CUSOLVER.gesv!(x, Ad, bd)
        x
    end,
    "Generated" => (Ad, bd) -> proposed_fn(Ad, bd)
)

# Store results for plotting
results = Dict()

for sparsity in sparsity_levels
    println("\n=== Sparsity: $sparsity ===")
    
    # Generate problem
    A = sprand(N, N, sparsity)
    b = rand(N)
    Ad = CuArray(Matrix(A))
    bd = CuArray(b)
    
    results[sparsity] = Dict()
    
    for (solver_name, solver_fn) in solvers
        println("  $solver_name...")
        
        # Warm-up
        bd_warm = CuArray(copy(b))
        try
            x_warm = solver_fn(Ad, bd_warm)
            CUDA.synchronize()
        catch e
            println("    Warning: solver failed during warm-up: $e")
            continue
        end
        
        # Benchmark
        bd_bench = CuArray(copy(b))
        try
            bench = @benchmark begin
                x = $(solver_fn)($Ad, $bd_bench)
                CUDA.synchronize()
            end seconds = 5 samples = 10
            
            time_ms = median(bench.times) / 1e9  # Convert to s
            
            # Compute error
            bd_err = CuArray(copy(b))
            x_sol = solver_fn(Ad, bd_err)
            CUDA.synchronize()
            error = norm(Ad*x_sol - bd_err) / norm(bd_err)
            
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
    legend=:bottomright,
    xlabel="Time (s)",
    ylabel="Relative residual: ||Ax - b||₂ / ||b||₂",
 #   xscale=:log10,
    yscale=:log10,
    guidefontsize=22,#18,
    tickfontsize=20, #16,
    legendfontsize=18, #14,
    margin=5*Plots.mm,
    framestyle=:box,
    title="Random Matrices of Size $(N)x$(N),\n Varying Sparsity Levels (ρ) and\n GPU Solvers",
    titlefontsize=22,
)

# Symbols encode sparsity levels; colors encode solvers.
# Define marker for each sparsity and a color for each solver.
## Sparsity shapes
marker_map_sparsity = OrderedDict(0.1=>:circle, 0.5=>:square, 0.9=>:utriangle)
## Solver color shades
color_map_solver = OrderedDict("Default"=>:red, "gesv!"=>:blue, "Generated"=>:green)

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