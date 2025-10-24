function proposed_fn(A, b)
    m, n = size(A)
    # Validate right-hand side dimensions
    if isa(b, AbstractVector)
        length(b) == m || throw(DimensionMismatch("length(b) != size(A,1)"))
    elseif isa(b, AbstractMatrix)
        size(b,1) == m || throw(DimensionMismatch("size(b,1) != size(A,1)"))
    else
        throw(ArgumentError("b must be a vector or matrix"))
    end

    if m == n
        # Square system: LU factorization
        F = lu(A)  # dense -> DenseLU, sparse -> SparseLU
        # For sparse factorizations, F \ b is robust and fast; for dense,
        # use the stored L,U and permutation for an explicit two-stage solve.
        if !issparse(A) && hasproperty(F, :L) && hasproperty(F, :U) && hasproperty(F, :p)
            p = F.p
            if isa(b, AbstractVector)
                y = F.L \ b[p]
                x = F.U \ y
                return x
            else
                y = F.L \ b[p, :]
                x = F.U \ y
                return x
            end
        else
            return F \ b
        end
    else
        # Rectangular: use QR for least-squares or minimum-norm solution
        F = qr(A)
        return F \ b
    end
end