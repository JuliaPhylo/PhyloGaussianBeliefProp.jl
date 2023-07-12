function marginalizebelief(h,J,g, keep_index)
    integrate_index = setdiff(1:length(h), keep_index)
    marginalizebelief(h,J,g, keep_index, integrate_index)
end
function marginalizebelief(h,J,g, keep_index, integrate_index)
    isempty(integrate_index) && return (h,J,g)
    ni = length(integrate_index)
    Ji = PDMat(view(J, integrate_index, integrate_index)) # fails if cholesky fails, e.g. if J=0
    Jk = view(J, keep_index, keep_index)
    Jki = view(J, integrate_index, keep_index)
    hi = view(h, integrate_index)
    hk = view(h, keep_index)
    messageJ = Jk - X_invA_Xt(Ji, Jki) # Jk - Jki Ji^{-1} Jki'
    μi = Ji \ hi
    messageh = hk - Jki * μi
    messageg = g + (ni*log2π + LA.logdet(Ji) + sum(hi .* μi))/2
    return (messageh, messageJ, messageg)
end

"""
    absorbevidence!(h,J,g, dataindex, datavalues)

Absorb evidence, at indices `dataindex` and using `datavalues`.
Warnings:
- a subset of `h` is modified in place
- traits are assumed to come in the same order in `dataindex` as in `datavalues`.
"""
function absorbevidence!(h,J,g, dataindex, datavalues)
    numt = length(datavalues)
    length(dataindex) == numt || error("data values and indices have different numbers of traits")
    hasdata = .!ismissing.(datavalues)
    absorb_ind = dataindex[hasdata]
    nvar = length(h)
    keep_ind = setdiff(1:nvar, absorb_ind)
    # index of variables with missing data, after removing variables with data:
    missingdata_indices = indexin(dataindex[.!hasdata], keep_ind)
    data_nm = view(datavalues, hasdata) # non-missing data values
    length(absorb_ind) + length(keep_ind) == nvar ||
        error("data indices go beyond belief size")
    Jkk = view(J, keep_ind,   keep_ind) # avoid copying
    Jka = view(J, keep_ind, absorb_ind)
    Jdata = Jka * data_nm
    g  += sum(view(h, absorb_ind) .* data_nm) - sum(Jdata .* data_nm)/2
    hk = view(h, keep_ind) .- Jdata # modifies h in place for a subset of indices
    return hk, Jkk, g, missingdata_indices
end

"""
    absorbleaf!(h,J,g, rowindex, columntable)

Absorb evidence from a leaf, given in `col[rowindex]` of each column in the table,
then marginalizes out any variable for a missing trait at that leaf.
See [`absorbevidence!`](@ref) and [`marginalizebelief`](@ref).
Warning:
The leaf traits are assumed to correspond to the first variables in `h` (and `J`),
as is output by [`factor_treeedge`](@ref).
"""
function absorbleaf!(h,J,g, rowindex, tbl)
    datavalues = [col[rowindex] for col in tbl]
    h,J,g,missingindices = absorbevidence!(h,J,g, 1:length(datavalues), datavalues)
    if !isempty(missingindices)
        h,J,g = marginalizebelief(h,J,g, setdiff(1:length(h), missingindices), missingindices)
    end
    # fixit: check that we correctly remove the missing data from scope
    return h,J,g
end
