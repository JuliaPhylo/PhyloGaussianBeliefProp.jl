"""
    marginalizebelief(belief::AbstractBelief, keep_index)
    marginalizebelief(h,J,g, keep_index)
    marginalizebelief(h,J,g, keep_index, integrate_index)

Canonical form (h,J,g) of the input belief, in which variables at indices
`keep_index` have been integrated out. If we use `I` and `S` subscripts
to denote subvectors and submatrices at indices to integrate out
(I: `integrate_index`) and indices to keep (S: save for sepset, `keep_index`)
then the returned belief parameters are:

``h_S - J_{S,I} J_I^{-1} h_I``

``J_S - J_{S,I} J_I^{-1} J_{I,S}``

and

``g + (\\log|2\\pi J_I^{-1}| + h_I^{T} J_I^{-1} h_I)/2 .``

These operations fail if the Cholesky decomposition of ``J_I`` fails.
"""
marginalizebelief(b::AbstractBelief, keepind) =
    marginalizebelief(b.h, b.J, b.g[1], keepind)
function marginalizebelief(h,J,g::Real, keep_index)
    # todo: if isempty(keep_index) call different function to avoid work for an empty messageJ
    integrate_index = setdiff(1:length(h), keep_index)
    marginalizebelief(h,J,g, keep_index, integrate_index)
end
function marginalizebelief(h,J,g::Real, keep_index, integrate_index)
    isempty(integrate_index) && return (h,J,g)
    ni = length(integrate_index)
    Ji = PDMat(view(J, integrate_index, integrate_index)) # fails if cholesky fails, e.g. if J=0
    Jk  = view(J, keep_index, keep_index)
    Jki = view(J, keep_index, integrate_index)
    hi = view(h, integrate_index)
    hk = view(h, keep_index)
    messageJ = Jk - X_invA_Xt(Ji, Jki) # Jk - Jki Ji^{-1} Jki'
    μi = Ji \ hi
    messageh = hk - Jki * μi
    messageg = g + (ni*log2π - LA.logdet(Ji) + sum(hi .* μi))/2
    return (messageh, messageJ, messageg)
end

"""
    integratebelief!(belief::AbstractBelief)
    integratebelief(h,J,g)

(μ,g) from fully integrating the belief, that is:
``μ = J^{-1} h`` and
``g + (\\log|2\\pi J^{-1}| + h^{T} J^{-1} h)/2 .``
The first form updates `belief.μ`.
"""
function integratebelief!(b::AbstractBelief)
    μ, g = integratebelief(b.h, b.J, b.g[1])
    b.μ .= μ
    return (μ,g)
end
function integratebelief(h,J,g::Real)
    ni = length(h)
    Ji = PDMat(J) # fails if cholesky fails, e.g. if J=0
    μi = Ji \ h
    messageg = g + (ni*log2π - LA.logdet(Ji) + sum(h .* μi))/2
    return (μi, messageg)
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
    Jkk = view(J, keep_ind,     keep_ind) # avoid copying
    Jk_data = view(J, keep_ind,   absorb_ind) * data_nm
    Ja_data = view(J, absorb_ind, absorb_ind) * data_nm
    g  += sum(view(h, absorb_ind) .* data_nm) - sum(Ja_data .* data_nm)/2
    hk = view(h, keep_ind) .- Jk_data # modifies h in place for a subset of indices
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
