"""
    marginalizebelief(belief::AbstractBelief, keep_index)
    marginalizebelief(h,J,g, keep_index)
    marginalizebelief(h,J,g, keep_index, integrate_index)

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
    Ji = PDMat(view(J, integrate_index, integrate_index)) # fails if non positive definite, e.g. J=0
    Jk  = view(J, keep_index, keep_index)
    Jki = view(J, keep_index, integrate_index)
    hi = view(h, integrate_index)
    hk = view(h, keep_index)
    messageJ = Jk - X_invA_Xt(Ji, Jki) # Jk - Jki Ji^{-1} Jki' without inv(Ji)
    μi = Ji \ hi
    messageh = hk - Jki * μi
    messageg = g + (ni*log2π - LA.logdet(Ji) + LA.dot(hi, μi))/2
    return (messageh, messageJ, messageg)
end

"""
    defaultmessage(cluster_to::AbstractBelief, d::Integer)

Canonical form (Δh, ΔJ, Δg) of a default received message for `cluster_to`
(received messages are represented as (Δh, ΔJ, Δg), and sent messages as
(h, J, g)). `d` is the dimension of the message, and is determined by the
dimension of the sepset (i.e. no. of traits × trait dimension) through which the
message is sent.

Δh is a length-`d` vector of zeroes, g is 0, and J is a `d`-by-`d` identity
matrix scaled by ϵ, where ϵ is set adaptively based on `cluster_to`.
"""
function defaultmessage(cluster_to::AbstractBelief, d::Integer)
    #= Other options tried:
    (1) ϵ = 1.0
    (2) ϵ = eps() # machine epsilon ≈ e-16
    (3) ϵ = minimum(abs, cluster_to.J)

    Option (1) works for the existing test cases ("test_calibration.jl") when
    default messages are generated on the fly (withdefault=true in
    `propagate_belief!`) whenever the sending-cluster's belief is degenerate
    (semi-definite) so that marginalization produces ∞

    Options (1) - (3) each fail for one or more of the existing test cases
    when we do not generate default messages on the fly and instead initialize
    the cluster/sepset beliefs (`init_messages!`) so that all subsequent
    messages (i.e. sepset belief candidates) are positive-definite. In this case,
    bad choice of ϵ wrt cluster_to.J (e.g. ϵ=1.0 >> entries of cluster_to.J in
    magnitude, which can happen if a very large σ² is tried out during
    optimization) can introduce rounding errors, semi-definite messages, and
    degenerate beliefs.
    =#
    #= fixit: happens to work for existing test cases, but needs to be more
    robust if we are opting NOT to generate default messages on the fly =#
    ϵ = maximum(abs, cluster_to.J)
    return (zeros(d), ϵ*LA.I(d), 0.0)
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
        h,J,g = marginalizebelief(h,J,g, setdiff(1:length(h), missingindices), missingindices)
    end
    return h,J,g
end

"""
    propagate_belief!(cluster_to, sepset, cluster_from, withdefault::Bool=true)
    propagate_belief!(cluster_to, sepset)

Update the canonical parameters of the beliefs in `cluster_to` and in `sepset`,
by marginalizing the belief in `cluster_from` to the sepset's variable and
passing that message.
A tuple, whose first element is `sepset` (after its canonical parameters have
been updated), and whose second element is a tuple
(Δh :: AbstractVector{<:Real}, ΔJ :: AbstractMatrix{<:Real}) representing the
residual between the canonical parameters of the current message from `cluster_to`
to `cluster_from` and the previous sepset belief (i.e. before updating).

The second form sends a default message to `cluster_to`, through `sepset` (it is
assumed that these are adjacent). While such messages always preserve the cluster
graph invariant, they can restrict the set of feasible message schedules if
applied injudiciously. [`regularizebeliefs!`](@ref) should avoid the need for this.

Warning: only the `h`, `J` and `g` parameters are updated, not `μ`.
Does not check that `cluster_from` and `cluster_to` are of cluster type,
or that `sepset` is of sepset type, but does check that the labels and scope
of `sepset` are included in each cluster.
"""
function propagate_belief!(cluster_to::AbstractBelief, sepset::AbstractBelief,
        cluster_from::AbstractBelief, withdefault::Bool=false)
    #= fixit: discuss the `withdefault` option. Should there be an option? If
    so, then methods that eventually call `propagate_belief!` have to be
    modified to pass this flag (e.g. `calibrate!`,
    `propagate_1traversal_postorder!`, `propagate_1traversal_preorder!`) =#
    # 1. compute message: marginalize cluster_from to variables in sepset
    #    requires cluster_from.J[keep,keep] to be invertible
    # `keepind` can be empty (e.g. if `cluster_from` is entirely "clamped")
    keepind = scopeindex(sepset, cluster_from)
    # canonical parameters of message received by `cluster_to`
    # message sent: (h, J, g), message received: (Δh, ΔJ, Δg)
    Δh, ΔJ, Δg = try
        h, J, g = marginalizebelief(cluster_from, keepind)
        # `cluster_from` is nondegenerate wrt the variables to be integrated out
        (h .- sepset.h, J .- sepset.J, g - sepset.g[1])
    catch ex
        isa(ex, LA.PosDefException) && withdefault || throw(ex)
        # `cluster_from` is degenerate so `cluster_to` receives a default message
        defaultmessage(cluster_to, length(keepind))
    end
    upind = scopeindex(sepset, cluster_to) # indices to be updated
    # 2. extend message to scope of cluster_to and propagate
    view(cluster_to.h, upind)        .+= Δh
    view(cluster_to.J, upind, upind) .+= ΔJ
    cluster_to.g[1]                   += Δg
    # 3. update sepset belief
    sepset.h   .+= Δh
    sepset.J   .+= ΔJ
    sepset.g[1] += Δg
    return sepset, (ΔJ, Δh, Δg)
end
function propagate_belief!(cluster_to::AbstractBelief, sepset::AbstractBelief)
    upind = scopeindex(sepset, cluster_to)
    isempty(upind) && return
    _, ΔJ, _ = defaultmessage(cluster_to, length(upind))
    view(cluster_to.J, upind, upind) .+= ΔJ # update cluster_to belief
    sepset.J .+= ΔJ # update sepset belief
    return
end
