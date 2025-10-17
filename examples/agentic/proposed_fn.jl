function proposed_fn(a)
    n = length(a)
    if n <= 1
        return a
    end
    mid = fld(n, 2)
    left = proposed_fn(view(a, 1:mid))
    right = proposed_fn(view(a, mid+1:n))
    out = similar(a)
    i = 1; j = 1; k = 1
    llen = length(left); rlen = length(right)
    while i <= llen && j <= rlen
        if left[i] <= right[j]
            out[k] = left[i]
            i += 1
        else
            out[k] = right[j]
            j += 1
        end
        k += 1
    end
    while i <= llen
        out[k] = left[i]
        i += 1
        k += 1
    end
    while j <= rlen
        out[k] = right[j]
        j += 1
        k += 1
    end
    return out
end