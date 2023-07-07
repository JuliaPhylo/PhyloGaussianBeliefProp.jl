# absorb evidence at leaf: tbl[v][row] for variable v
# warning: assumes *ordered* indices
function marginalizebelief(μ,h,J,g, keep_index)
    integrate_index = setdiff(1:n, keep_index)
    marginalizebelief(μ,h,J,g, keep_index, integrate_index)
end
function marginalizebelief(μ,h,J,g, keep_index, integrate_index)
    Ji = view(J, integrate_index, integrate_index)
    Jk = view(J, keep_index, keep_index)
    Jik = view(J, integrate_index, keep_index)
    hi = view(h, integrate_index)
    hk = view(h, keep_index)
    Vi = inv(Ji)
end
function absorbevidence(μ,h,J,g, evidence_index, row, tbl)
end
