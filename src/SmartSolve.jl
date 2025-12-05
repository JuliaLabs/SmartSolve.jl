module SmartSolve

using MatrixDepot
using LinearAlgebra
using DataFrames
using OrderedCollections
using CSV
using CairoMakie
using DecisionTree
using Random
using BenchmarkTools
using BSON
using SparseArrays
using OpenAI
using CUDA

include("SmartDiscovery.jl")
include("SmartDB.jl")
include("SmartModel.jl")
include("Utils.jl")
include("Agentic.jl")
include("test_performance.jl")
# include("test_performance_cuda.jl")

export generate_default_code, gen_linear_solver, gen_linear_solver_cuda, printhist

end # module SmartSolve
