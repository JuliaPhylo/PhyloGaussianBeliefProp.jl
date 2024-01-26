"""
    BPPosDefException

Exception thrown when a belief message cannot be computed, that is, when the
submatrix of the precision `J`, subsetted to the variables to be integrated out,
is not [positive definite](https://en.wikipedia.org/wiki/Definiteness_of_a_matrix).
It has a message `msg` field (string), and an `info` field (integer) inherited
from `LinearAlgebra.PosDefException`, to indicate the location of (one of)
the eigenvalue(s) which is (are) less than/equal to 0.
"""
struct BPPosDefException <: Exception
    msg::String
    info::LA.BlasInt
end
function Base.showerror(io::IO, ex::BPPosDefException)
    print(io, "BPPosDefException: $(ex.msg)\nmatrix is not ")
    if ex.info == -1
        print(io, "Hermitian.")
    else
        print(io, "positive definite.")
    end
end

"""
    marginalizebelief(belief::AbstractBelief, keep_index)
    marginalizebelief(h,J,g, keep_index, beliefmetadata)
    marginalizebelief(h,J,g, keep_index, integrate_index, beliefmetadata)

Canonical form (h,J,g) of the input belief, after all variables except those at
indices `keep_index` have been integrated out. If we use `I` and `S` subscripts
to denote subvectors and submatrices at indices to integrate out
(I: `integrate_index`) and indices to keep (S: save for sepset, `keep_index`)
then the returned belief parameters are:

``h_S - J_{S,I} J_I^{-1} h_I``

``J_S - J_{S,I} J_I^{-1} J_{I,S}``

and

``g + (\\log|2\\pi J_I^{-1}| + h_I^{T} J_I^{-1} h_I)/2 .``

These operations fail if the Cholesky decomposition of ``J_I`` fails.
In that case, an error of type [`BPPosDefException`](@ref) is thrown
with a message about the `beliefmetadata`,
which can be handled by downstream functions.
"""
marginalizebelief(b::AbstractBelief, keepind) =
    marginalizebelief(b.h, b.J, b.g[1], keepind, b.metadata)
function marginalizebelief(h,J,g::Real, keep_index, metadata)
    # todo: if isempty(keep_index) call different function to avoid work for an empty messageJ
    integrate_index = setdiff(1:length(h), keep_index)
    marginalizebelief(h,J,g, keep_index, integrate_index, metadata)
end
function marginalizebelief(h,J,g::Real, keep_index, integrate_index, metadata)
    isempty(integrate_index) && return (h,J,g)
    ni = length(integrate_index)
    Ji = view(J, integrate_index, integrate_index)
    Jk  = view(J, keep_index, keep_index)
    Jki = view(J, keep_index, integrate_index)
    hi = view(h, integrate_index)
    hk = view(h, keep_index)
    # Ji = Jki = hi = 0 if missing data: fake issue
    ϵ = eps(eltype(J))
    if all(isapprox.(Ji, 0, atol=ϵ)) && all(isapprox.(hi, 0, atol=ϵ)) && all(isapprox.(Jki, 0, atol=ϵ))
        return (hk, Jk, g)
    end
    Ji = try # re-binds Ji
        PDMat(Ji) # fails if non positive definite, e.g. Ji=0
    catch pdmat_ex
        if isa(pdmat_ex, LA.PosDefException)
            ex = BPPosDefException("belief $metadata, integrating $(integrate_index)", pdmat_ex.info)
            throw(ex)
        else
            rethrow(pdmat_ex)
        end
    end
    messageJ = Jk - X_invA_Xt(Ji, Jki) # Jk - Jki Ji^{-1} Jki' without inv(Ji)
    μi = Ji \ hi
    messageh = hk - Jki * μi
    messageg = g + (ni*log2π - LA.logdet(Ji) + LA.dot(hi, μi))/2
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
function integratebelief(h,J,g)
    Ji = PDMat(J) # fails if cholesky fails, e.g. if J=0
    integratebelief(h,Ji,g)
end
function integratebelief(h,J::Union{LA.Cholesky{T},PDMat{T}},g::T) where T<:Real
    n = length(h)
    μ = J \ h
    messageg = g + (n*T(log2π) - LA.logdet(J) + sum(h .* μ))/2
    return (μ, messageg)
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
    if isempty(absorb_ind)
        return h, Jkk, g, missingdata_indices
    end
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
        @debug "leaf data $(join(datavalues,',')), J=$(round.(J, digits=2)), will integrate at index $(join(missingindices,','))"
        h,J,g = marginalizebelief(h,J,g, setdiff(1:length(h), missingindices), missingindices, "leaf row $rowindex")
    end
    return h,J,g
end

"""
    propagate_belief!(cluster_to, sepset, cluster_from, residual)

Update the canonical parameters of the beliefs in `cluster_to` and in `sepset`,
by marginalizing the belief in `cluster_from` to the sepset's variable and
passing that message.
The change in sepset belief (`Δh` and `ΔJ`: new - old) is stored in `residual`.

## Degeneracy

Propagating a belief requires the `cluster_from` belief to have a
non-degenerate `J_I`: submatrix of `J` for the indices to be integrated out.
Problems arise if this submatrix has one or more 0 eigenvalues, or infinite values
(see [`marginalizebelief`](@ref)).
If so, a [`BPPosDefException`](@ref) is returned **but not thrown**.
Downstream functions should try & catch these failures, and decide how to proceed.
See [`regularizebeliefs_bycluster!`](@ref) to reduce the prevalence of degeneracies.

## Output

- `nothing` if the message was calculated with success
- a [`BPPosDefException`](@ref) object if marginalization failed. In this case,
  *the error is not thrown*: downstream functions should check for failure
  (and may choose to throw the output error object).

## Warnings

- only the `h`, `J` and `g` parameters are updated, not `μ`.
- Does not check that `cluster_from` and `cluster_to` are of cluster type,
  or that `sepset` is of sepset type, but does check that the labels and scope
  of `sepset` are included in each cluster.
"""
function propagate_belief!(
    cluster_to::AbstractBelief,
    sepset::AbstractBelief,
    cluster_from::AbstractBelief,
    residual::AbstractResidual
)
    # 1. compute message: marginalize cluster_from to variables in sepset
    #    requires cluster_from.J[I,I] to be invertible, I = indices other than `keepind`
    #    marginalizebelief sends BPPosDefException otherwise.
    # `keepind` can be empty (e.g. if `cluster_from` is entirely "clamped")
    keepind = scopeindex(sepset, cluster_from)
    h,J,g = try marginalizebelief(cluster_from, keepind)
    catch ex
        isa(ex, BPPosDefException) && return ex # output the exception: not thrown
        rethrow(ex) # exception thrown if other than BPPosDefException
    end
    # calculate residual
    residual.Δh .= h .- sepset.h
    residual.ΔJ .= J .- sepset.J
    # 2. extend message to scope of cluster_to and propagate
    upind = scopeindex(sepset, cluster_to) # indices to be updated
    view(cluster_to.h, upind)        .+= residual.Δh
    view(cluster_to.J, upind, upind) .+= residual.ΔJ
    cluster_to.g[1]                   += g  - sepset.g[1]
    # 3. update sepset belief
    sepset.h   .= h
    sepset.J   .= J
    sepset.g[1] = g
    return nothing
end
