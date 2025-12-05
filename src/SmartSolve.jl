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
using Dagger

include("SmartDiscovery.jl")
include("SmartDB.jl")
include("SmartModel.jl")
include("Utils.jl")
include("Agentic.jl")

export generate_default_code, gen_linear_solver, gen_linear_solver_cuda, gen_linear_solver_dagger, printhist

end # module SmartSolve
