function marginalizebelief(h,J,g, keep_index)
    integrate_index = setdiff(1:n, keep_index)
    marginalizebelief(h,J,g, keep_index, integrate_index)
end
function marginalizebelief(h,J,g, keep_index, integrate_index)
    ni = length(integrate_index)
    Ji = PDMat(view(J, integrate_index, integrate_index)) # fails if cholesky fails, e.g. if J=0
    Jk = view(J, keep_index, keep_index)
    Jki = view(J, integrate_index, keep_index)
    hi = view(h, integrate_index)
    hk = view(h, keep_index)
    messageJ = Jk - X_invA_Xt(Ji, Jki) # Jk - Jki Ji^{-1} Jki'
    μi = Ji \ hi
    messageh = hk - Jki * μi
    messageg = g + (ni*log2π + lodet(Ji) + sum(hi .* μi))/2
    return (messageh, messageJ, messageg)
end

# absorb evidence at leaf: tbl[v][row] for variable v
# warning: assumes *ordered* indices
function absorbevidence(μ,h,J,g, evidence_index, row, tbl)
    # todo: find indices with no data
    # marginalized variables (indices) with no data
    # plug in observed values
end
