# Benchmark and generate linear solver

Automatically generate an optimized LU decomposition by selecting a specialized solver tailored to the input matrix structure.

Define candidate algorithms
```julia
dgetrf(A::Matrix) = lu(A)
dgetrf(A::SparseMatrixCSC) = lu(Matrix(A))
dgetrf(A::SparseMatrixCSC{Bool, Int64}) = lu(Matrix(A))
dgetrf(A::Symmetric) = lu(A.data)
bandedlu(A::Matrix) = BandedMatrices.lu(BandedMatrix(sparse(A)))
bandedlu(A::SparseMatrixCSC{Float64, Int64}) = BandedMatrices.lu(BandedMatrix(A))
bandedlu(A::SparseMatrixCSC{Int64, Int64}) = BandedMatrices.lu(BandedMatrix(Float64.(A)))
bandedlu(A::SparseMatrixCSC{Bool, Int64}) = BandedMatrices.lu(BandedMatrix(Float64.(A)))
bandedlu(A::Symmetric) = BandedMatrices.lu(Float64.(BandedMatrix(sparse(A.data))))
algs = [dgetrf, bandedlu]
```

Define your custom matrices to be included in training
```julia
n = 2^12;
A = ...
B = ...
mats = [A, B]
```

Generate a smart version of the algorithm
```julia
alg_name  = "lu"
alg_path = "smartlu/"
smartsolve(alg_path, alg_name, algs; n_experiments = 1,
           mats = mats, ns = [2^8.2^10,2^12], features = [:isbandedpattern])
```

Include and run the newly generated algorithm
```julia
include("$alg_path/smart$alg_name.jl")
smartlu(A)\b
```
