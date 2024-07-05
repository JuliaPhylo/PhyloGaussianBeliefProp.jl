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
    marginalize(belief::AbstractFactorBelief, keep_index)
    marginalize(h,J,g, keep_index, beliefmetadata)
    marginalize(h,J,g, keep_index, integrate_index, beliefmetadata)

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
marginalize(b::AbstractFactorBelief, keepind) =
    marginalize(b.h, b.J, b.g[1], keepind, b.metadata)
function marginalize(h,J,g::Real, keep_index, metadata)
    integrate_index = setdiff(1:length(h), keep_index)
    marginalize(h,J,g, keep_index, integrate_index, metadata)
end
function marginalize(h,J,g::Real, keep_index, integrate_index, metadata)
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
    marginalize!(cluster_from::GeneralizedBelief, keep_index)

Marginalize a generalized belief, accessed from `cluster_from`, by integrating out all
variables at indices `keep_index`. The resulting marginal is stored in the message fields of
`cluster_from`.
"""
function marginalize!(cluster_from::GeneralizedBelief, keepind)
    mm = length(keepind) # dimension for marginal
    m1 = size(cluster_from.Q)[1]
    k = cluster_from.k[1] # k ≤ m1
    Q1 = cluster_from.Q[keepind,1:(m1-k)] # mm x (m1-k)
    R1 = cluster_from.R[keepind,1:k] # mm x k
    Λ = LA.Diagonal(view(cluster_from.Λ,1:(m1-k))) # matrix, not vector
    
    ## compute marginal and save parameters to cluster_from buffer
    # constraint rank (todo: set threshold for a zero singular value)
    U1, S1, V1 = LA.svd(Q1; full=true) # U1: mm x mm, S1: min(mm,m1-k) x 1, V1: (m1-k) x (m1-k)
    nonzeroind = findall(S1 .!= 0.0) # indices for non-zero singular values
    zeroind = setdiff(1:(m1-k), nonzeroind) # indices for zero singular values
    km = mm - length(nonzeroind)
    cluster_from.kmsg[1] = km
    # constraint
    cluster_from.Rmsg[1:mm,1:km] = view(U1,:,setdiff(1:mm,nonzeroind)) # mm x km
    cluster_from.cmsg[1:km] = transpose(view(cluster_from.Rmsg,1:mm,1:km))*R1*
        view(cluster_from.c,1:k)
    # precision
    V = V1[:,zeroind] # nullspace(Q1): (m1-k) x (m1-k-mm+km)
    W = transpose(R1)*Q1 # k x (m1-k)
    if !isempty(W)
        # orthogonalize W is non-empty
        # note: qr can be run if input matrix has col dim 0, but not row dim 0
        W = LA.qr(W)
        W = W.Q[:,findall(LA.diag(W.R) .!== 0.0)]
    end
    Q2 = cluster_from.Q[setdiff(1:m1,keepind),1:(m1-k)] # (m1-mm) x (m1-k)
    R2 = cluster_from.R[setdiff(1:m1,keepind),1:k] # (m1-mm) x k
    F = transpose(W*((R2*W)\Q2)) # transpose(W(R2*W)⁺Q2): (m1-k) x k
    G = (transpose(Q1)-F*transpose(R1))*view(U1,:,nonzeroind) # (m1-k) x (mm-km)
    S = V*((transpose(V)*Λ*V) \ transpose(V)) # (m1-k) x (m1-k)
    Z, Λm = LA.svd(transpose(G)*(Λ-Λ*S*Λ)*G)
    cluster_from.Λmsg[1:(mm-km)] = Λm
    cluster_from.Qmsg[1:mm,1:(mm-km)] = view(U1,:,nonzeroind)*Z
    # potential
    Fc = F*cluster_from.c[1:k] # (m1-k) x 1
    ΛFc = Λ*Fc # (m1-k) x 1
    h_ΛFc = cluster_from.h[1:(m1-k)] - ΛFc
    cluster_from.hmsg[1:(mm-km)] = transpose(Z)*transpose(G)*(LA.I-Λ*S)*(h_ΛFc)
    # normalization constant
    cluster_from.gmsg[1] = cluster_from.g[1] + transpose(h_ΛFc+0.5*ΛFc)*Fc
    if m1 != k # Λ: (m1-k) x (m1-k) not empty
        # if Λ empty but V is not, then VᵀΛV defaults to a zero matrix with ∞ logdet!
        cluster_from.gmsg[1] -= 0.5*LA.logdet((1/2π)*transpose(V)*Λ*V)
    end
    if k > 0 # R2ᵀR2: k x k not empty
        cluster_from.gmsg[1] -= 0.5*LA.logdet(transpose(R2*W)*R2*W)
    end
    return nothing
end

"""
    integratebelief!(belief::CanonicalBelief)
    integratebelief!(belief::GeneralizedBelief)
    integratebelief(h,J,g)

(μ, norm) from fully integrating the belief. The first two forms update
`belief.μ`.

For a canonical belief, μ = J⁻¹h and norm = g + (log|2πJ⁻¹| + hᵀJ⁻¹h)/2. μ is
the mean of x, where x is the scope of `belief`. norm is the normalization
constant from integrating out x.

For a generalized belief, μ = QΛ⁻¹h and norm = g + (log|2πΛ⁻¹| + hᵀΛ⁻¹h)/2.
In this case, μ is the mean of QQᵀx and norm is the normalization constant from
integrating out Qᵀx. If Q is square, then the interpretation of μ and norm is
the same as for a canonical belief.
"""
function integratebelief!(b::CanonicalBelief)
    μ, norm = integratebelief(b.h, b.J, b.g[1])
    b.μ[:] = μ
    return (μ, norm)
end
function integratebelief!(b::GeneralizedBelief)
    m = size(b.Q)[1]
    k = b.k[1]
    #= 𝒟(x;Q,R,Λ,h,c,g) = exp(-xᵀ(QΛQᵀ)x/2+(Qh)ᵀx+g)⋅δ(Rᵀx-c)
    Take Qᵀx ∼ 𝒩(Λ⁻¹h,Λ⁻¹):
    (1) μ = E[QQᵀx] = QΛ⁻¹h
    (2) messageg = ∫C(Qᵀx;Λ,h,g)d(Qᵀx) = exp(g)⋅exp(log|2πΛ⁻¹| + hᵀΛ⁻¹h)/2) =#
    # todo: check that b.Λ[1:(m-k)] has no 0-entries so that C(Qᵀx;Λ,h,g) is normalizable
    μ = view(b.Q,:,1:(m-k))*(view(b.h,1:(m-k)) ./ view(b.Λ,1:m-k))
    messageg = b.g[1] + (m*log(2π) - log(prod(view(b.Λ,1:(m-k)))) +
        sum(view(b.h,1:(m-k)) .^2 ./ view(b.Λ,1:(m-k))))/2
    b.μ[1:(m-k)] = μ
    return (μ, messageg)
end
function integratebelief(h,J,g)
    # Ji = PDMat(J) # fails if cholesky fails, e.g. if J=0
    Ji = PDMat(LA.Symmetric(J)) # todo: discuss enforcing J to be symmetric
    integratebelief(h,Ji,g)
end
function integratebelief(h,J::Union{LA.Cholesky{T},PDMat{T}},g::T) where T<:Real
    n = length(h)
    μ = J \ h
    norm = g + (n*T(log2π) - LA.logdet(J) + sum(h .* μ))/2
    return (μ, norm)
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
See [`absorbevidence!`](@ref) and [`marginalize`](@ref).
Warning:
The leaf traits are assumed to correspond to the first variables in `h` (and `J`),
as is output by [`factor_treeedge`](@ref).
"""
function absorbleaf!(h,J,g, rowindex, tbl)
    datavalues = [col[rowindex] for col in tbl]
    h,J,g,missingindices = absorbevidence!(h,J,g, 1:length(datavalues), datavalues)
    if !isempty(missingindices)
        @debug "leaf data $(join(datavalues,',')), J=$(round.(J, digits=2)), will integrate at index $(join(missingindices,','))"
        h,J,g = marginalize(h,J,g, setdiff(1:length(h), missingindices), missingindices, "leaf row $rowindex")
    end
    return h,J,g
end

"""
    extend!(cluster_to::GeneralizedBelief, sepset)
    extend!(cluster_to::GeneralizedBelief, upind, Δh, ΔJ, Δg)
    extend!(cluster_to::GeneralizedBelief, upind, R)

Extend and match the scope of an incoming message to the scope of generalized
belief `cluster_to` so that multiplication (see [`mult!`](@ref)) can be applied.
The parameters of the incoming message are extended within the message fields of
`cluster_to`.

The first form applies when the incoming message is accessed from the message fields of
generalized belief `sepset`.
The second form applies when the incoming message has an exponential quadratic form with
canonical parameters Δh, ΔJ, Δg.
The third form applies when the incoming message is a deterministic factor with constraint
matrix R.
"""
function extend!(cluster_to::GeneralizedBelief, sepset::GeneralizedBelief)
    # indices in `cluster_to` inscope variables of `sepset` inscope variables
    upind = scopeindex(sepset, cluster_to)
    #=
    x,Q,Λ from sepset buffer are extended as follows:
    (1) x is extended to Px', where x' is formed by stacking x on top of the
    inscope variables of `cluster_to` that are not in x, and P is a permutation
    matrix determined by `upind`
    (2) Q is extended to Q' = P[Q 0; 0 I], where P is a permutation matrix
    (3) Λ is extended to Λ' = [Λ 0; 0 0]
    (4) R is extended to R' = P[R; 0]

    - Px' matches the scope of `cluster_to`
    - the additional rows and columns in Q',Λ',R' correspond to the variables
    present in x but not x'
    - Only Q,Λ,R, but not h,c need to be extended:
        - (Px')ᵀQ'Λ'Q'ᵀ(Px') = xᵀQΛQᵀx
        - hᵀQ'ᵀ(Px') = hᵀQᵀx
        - R'ᵀ(Px') = Rᵀx
    =#
    m1 = size(cluster_to.Q)[1]
    perm = [upind;setdiff(1:m1,upind)] # to permute rows
    m2 = size(sepset.Q)[1]
    # constraint rank
    k2 = sepset.kmsg[1]
    cluster_to.kmsg[1] = k2
    # constraint
    cluster_to.Rmsg[:,1:k2] .= 0.0 # reset buffer
    cluster_to.Rmsg[1:m2,1:k2] = view(sepset.Rmsg,:,1:k2)
    cluster_to.Rmsg[perm,:] = cluster_to.Rmsg[:,:] # permute rows of [R; 0]
    cluster_to.cmsg[1:k2] = view(sepset.cmsg,1:k2)
    # precision
    cluster_to.Qmsg .= 0.0
    cluster_to.Qmsg[LA.diagind(cluster_to.Qmsg)] .= 1.0 # 1s along the diagonal
    cluster_to.Qmsg[1:m2,1:(m2-k2)] = view(sepset.Qmsg,:,1:(m2-k2))
    cluster_to.Qmsg[perm,:] = cluster_to.Qmsg[:,:] # permute rows of [Q 0; 0 I]
    cluster_to.Λmsg .= 0.0
    cluster_to.Λmsg[1:(m2-k2)] = sepset.Λmsg[1:(m2-k2)]
    # potential
    cluster_to.hmsg .= 0.0
    cluster_to.hmsg[1:(m2-k2)] = view(sepset.hmsg,1:(m2-k2))
    # normalization constant
    cluster_to.gmsg[1] = sepset.gmsg[1]
    return
end
function extend!(cluster_to::GeneralizedBelief, upind, Δh, ΔJ, Δg)
    m1 = size(cluster_to.Q)[1]
    perm = [upind;setdiff(1:m1,upind)]
    m2 = size(ΔJ)[1]
    # constraint rank
    cluster_to.kmsg[1] = 0
    # cluster_to.Rmsg, cluster_to.cmsg not updated since constraint is irrelevant
    # precision
    Q, Λ = LA.svd(ΔJ)
    cluster_to.Qmsg .= 0.0
    cluster_to.Qmsg[LA.diagind(cluster_to.Qmsg)] .= 1.0
    cluster_to.Qmsg[1:m2,1:m2] = Q
    cluster_to.Qmsg[perm,:] = cluster_to.Qmsg[:,:]
    cluster_to.Λmsg .= 0.0
    cluster_to.Λmsg[1:m2] = Λ
    # potential
    cluster_to.hmsg .= 0.0
    cluster_to.hmsg[1:m2] = Q*Δh
    # normalization constant
    cluster_to.gmsg[1] = Δg
    return
end
function extend!(cluster_to::GeneralizedBelief, upind, R)
    # todo: checks that R is valid
    m1 = size(cluster_to.Q)[1]
    perm = [upind;setdiff(1:m1,upind)]
    m2, k2 = size(R)
    # contraint rank
    cluster_to.kmsg[1] = k2
    # constraint
    cluster_to.Rmsg[:,1:k2] .= 0
    cluster_to.Rmsg[1:m2,1:k2] = R
    cluster_to.Rmsg[perm,:] = cluster_to.Rmsg[:,:]
    cluster_to.cmsg[1:k2] .= 0
    # precision
    cluster_to.Qmsg .= 0
    cluster_to.Qmsg[LA.diagind(cluster_to.Qmsg)] .= 1.0
    cluster_to.Qmsg[1:m2,1:(m2-k2)] = LA.nullspace(transpose(R))
    cluster_to.Qmsg[perm,:] = cluster_to.Qmsg[:,:]
    cluster_to.Λmsg .= 0
    # potential
    cluster_to.hmsg .= 0
    # normalization constant
    cluster_to.gmsg[1] = 0
    return  
end

"""
    mult!(cluster_to::GeneralizedBelief, sepset)
    mult!(cluster_to::GeneralizedBelief, upind, Δh, ΔJ, Δg)
    mult!(cluster_to::GeneralizedBelief, upind, R)
    mult!(cluster_to::GeneralizedBelief)

Multiply a belief into the generalized belief `cluster_to`. The parameters of the incoming
message are extended (see [`extend!`](@ref)) to the scope of `cluster_to`'s parameters
within the message fields of `cluster_to` before multiplication is carried out.
The parameters of `cluster_to` are updated to those of the product.

The first method applies when the incoming message is accessed from the message fields of
generalized belief `sepset`.
The second method applies when the incoming message is an exponetial quadratic form with
canonical parameters ΔJ, Δh, Δg.
The third method applies when the incoming message is a deterministic factor with constraint
matrix R.
The fourth method is called by the previous methods, and performs multiplication directly
within `cluster_to`.
"""
function mult!(cluster_to::GeneralizedBelief, sepset::GeneralizedBelief)
    ## extend scope of incoming message and save parameters to cluster_to buffer
    extend!(cluster_to, sepset)
    ## multiply the beliefs (non-buffer and buffer) in cluster_to
    mult!(cluster_to)
end
function mult!(cluster_to::GeneralizedBelief, upind, Δh, ΔJ, Δg)
    extend!(cluster_to, upind, Δh, ΔJ, Δg)
    mult!(cluster_to)
    # cluster_to.h[upind] .+= residual.Δh
    # cluster_to[upind,upind] .+= residual.ΔJ
    # cluster_to.g[1] += g - sepset.g[1]
end
function mult!(cluster_to::GeneralizedBelief, upind, R)
    extend!(cluster_to, upind, R)
    mult!(cluster_to)
end
function mult!(cluster_to::GeneralizedBelief)
    # constraint rank
    k1 = cluster_to.k[1]
    k2 = cluster_to.kmsg[1]
    # contraints
    R = cluster_to.R
    R1 = R[:,1:k1]
    R2 = cluster_to.Rmsg[:,1:k2]
    c = cluster_to.c
    c1 = c[1:k1]
    c2 = cluster_to.cmsg[1:k2]
    # precisions
    m1 = size(cluster_to.Q)[1]
    Q1 = cluster_to.Q[:,1:(m1-k1)]
    Λ1 = cluster_to.Λ[1:(m1-k1)]
    m2 = size(cluster_to.Qmsg)[1]
    Q2 = cluster_to.Qmsg[:,1:(m2-k2)]
    Λ2 = cluster_to.Λmsg[1:(m2-k2)]
    # potentials
    h = cluster_to.h
    h1 = h[1:(m1-k1)]
    h2 = cluster_to.hmsg[1:(m2-k2)]

    ## compute product and save parameters to cluster_to
    # constraint
    V = LA.qr(Q1*transpose(Q1)*R2) # project R2 onto colsp(Q1)
    V = V.Q[:,findall(LA.diag(V.R) .!== 0.0)] # orthonormal basis for colsp(Q1*Q1ᵀ*R2)
    Δk1 = size(V)[2] # increase in constraint rank
    R[:,(k1+1):(k1+Δk1)] = V
    R2tV = transpose(R2)*V
    if k2 == Δk1 # transpose(R2)*V is square
        b = R2tV \ (c2 - transpose(R2)*R1*c1)
        lgdet = 2*LA.logabsdet(R2tV)[1] # equivalent to (log(abs(det(...))), sign(det(...)))
    else
        VtR2R2tV = transpose(R2tV)*R2tV
        b = (VtR2R2tV \ transpose(R2tV))*(c2 - transpose(R2)*R1*c1)
        lgdet = LA.logdet(VtR2R2tV)
    end
    c[(k1+1):(k1+Δk1)] = b
    # constraint rank
    k1 += Δk1
    cluster_to.k[1] = k1
    # for updating h and g (using new R1, c1, k1)
    Q1tVb = transpose(Q1)*V*b
    Λ1Q1tVb = Λ1.*Q1tVb
    Q2tR1c1 = transpose(Q2)*view(R,:,1:k1)*view(c,1:k1)
    Λ2Q2tR1c1 = Λ2.*Q2tR1c1
    # precision
    U = LA.nullspace(transpose(view(R,:,1:k1)))
    Z, Λ = LA.svd(transpose(U)*(Q1*LA.Diagonal(Λ1)*transpose(Q1) +
        Q2*LA.Diagonal(Λ2)*transpose(Q2))*U)
    cluster_to.Q[:,1:(m1-k1)] = U*Z
    cluster_to.Λ[1:(m1-k1)] = Λ
    # potential
    h[1:(m1-k1)] = transpose(view(cluster_to.Q,:,1:(m1-k1)))*(Q1*(h1-Λ1Q1tVb) +
        Q2*(h2-Λ2Q2tR1c1))
    # normalization constant
    cluster_to.g[1] += cluster_to.gmsg[1] + transpose(h1-0.5*Λ1Q1tVb)*Q1tVb +
        transpose(h2-0.5*Λ2Q2tR1c1)*Q2tR1c1 - 0.5*lgdet
    return nothing
end

"""
    divide!(sepset::GeneralizedBelief, cluster_from::GeneralizedBelief)
    divide!(sepset::CanonicalBelief, cluster_from::GeneralizedBelief)
    divide!(sepset::CanonicalBelief, h, J, g)

Divide an outgoing message (e.g. from `cluster_from`) by the belief of `sepset`.
If `sepset` is a generalized belief, then the quotient is stored in its message fields.
Otherwise, the quotient has an exponential quadratic form and its canonical parameters are
returned directly.

For the first two form, the dividend is accessed from the message field of `cluster_from`.
The third form applies when the dividend is a canonical belief with parameters (ΔJ, Δh, Δg).

Assumptions (not checked):
- Dividend and divisor are assumed to have matching scopes
- The constraint rank for the divisor cannot exceed that of the dividend
(i.e. cluster_from.kmsg[1] ≥ sepset.k[1]) for division to be well-defined
"""
function divide!(sepset::GeneralizedBelief, cluster_from::GeneralizedBelief)
    # constraint rank
    k1 = cluster_from.kmsg[1]
    k2 = sepset.k[1]
    # contraints
    m = size(sepset.Q)[1] # m ≥ max(k1,k2)
    R1 = cluster_from.Rmsg[1:m,1:k1]
    R2 = sepset.R[:,1:k2] # m x k2
    c1 = cluster_from.cmsg[1:k1]
    # precisions
    Q1 = cluster_from.Qmsg[1:m,1:(m-k1)]
    Λ1 = cluster_from.Λmsg[1:(m-k1)]
    Q2 = sepset.Q[:,1:(m-k2)]
    Λ2 = sepset.Λ[1:(m-k2)]
    # potentials
    h1 = cluster_from.hmsg[1:(m-k1)]
    h2 = sepset.h[1:(m-k2)]

    ## compute quotient and save parameters to sepset buffer
    # constraint rank
    k = k1-k2
    k ≥ 0 || error("quotient cannot have negative constraint rank")
    sepset.kmsg[1] = k
    # constraint
    cluster_from.Q[1:m,(m-k1+1):(m-k)] = R2 # use cluster_from.Q as a buffer for R2
    sepset.Rmsg[:,1:k] = LA.nullspace(view(cluster_from.Q,:,1:(m-k)))
    sepset.cmsg[1:k] = transpose(view(sepset.Rmsg,:,1:k))*R1*c1
    # precision
    Z, Λ = LA.svd(LA.Diagonal(Λ1)-transpose(Q1)*Q2*LA.Diagonal(Λ2)*transpose(Q2)*Q1)
    sepset.Λmsg .= 0.0 # reset buffer
    sepset.Λmsg[1:(m-k1)] = Λ
    sepset.Qmsg[:,1:(m-k1)] = Q1*Z
    sepset.Qmsg[:,(m-k1+1):(m-k)] = R2
    # potential
    Q2tR1c1 = transpose(Q2)*R1*c1
    Λ2Q2tR1c1 = Λ2.*Q2tR1c1
    sepset.hmsg[1:(m-k)] = (transpose(view(sepset.Qmsg,:,1:(m-k))) *
        (Q1*h1 - Q2*(h2 - Λ2Q2tR1c1)))
    # normalization constant
    sepset.gmsg[1] = cluster_from.gmsg[1] - sepset.g[1] -
        transpose(h2-0.5*Λ2Q2tR1c1)*Q2tR1c1
    
    ## update sepset (non-buffer) parameters to those of the dividend
    sepset.k[1] = k1
    sepset.R[:,1:k1] = R1
    sepset.c[1:k1] = c1
    sepset.Q[:,1:(m-k1)] = Q1
    sepset.Λ[1:(m-k1)] = Λ1
    sepset.h[1:(m-k1)] = h1
    sepset.g[1] = cluster_from.gmsg[1]
    return nothing
end
function divide!(sepset::CanonicalBelief, cluster_from::GeneralizedBelief)
    k1 = cluster_from.kmsg[1]
    m = size(sepset.J)[1]
    Q = cluster_from.Qmsg[1:m,1:(m-k1)]
    J = Q*LA.Diagonal(view(cluster_from.Λmsg,1:(m-k1)))*transpose(Q)
    h = Q*view(cluster_from.hmsg,1:(m-k1))
    return divide!(sepset, h, J, cluster_from.gmsg[1])
end
function divide!(sepset::CanonicalBelief, h, J, g)
    Δh = h .- sepset.h
    ΔJ = J .- sepset.J
    Δg = g - sepset.g[1]
    sepset.h[:] = h
    sepset.J[:] = J
    sepset.g[1] = g
    return Δh, ΔJ, Δg
end

"""
    propagate_belief!(cluster_to, sepset, cluster_from, residual)
    propagate_belief!(cluster_to, sepset, cluster_from, keepind)

Update the parameters of the beliefs `cluster_to` and `sepset` by marginalizing the belief
in `cluster_from` to the scope of `sepset` and passing that message.

The change in sepset belief (i.e. the quotient from dividing the outgoing message by the
current sepset belief) is stored in `residual`. We measure this "change" as the potential
and precision canonical parameters of the canonical part of the quotient (e.g. h and J if
the sepset is a canonical belief, or Qh and QΛQᵀ if it is a generalized belief).

The second method is only called by the first if `cluster_to`, `sepset`, `cluster_from` are
not all canonical beliefs, and dispatches on their combination of types (i.e. canonical or
generalized). In this event, there are 3 cases:
(1) `sepset`, `cluster_to` and `cluster_from` are all generalized beliefs
(2) `sepset` and `cluster_from` are canonical beliefs, and `cluster_to` is a generalized
belief
(3) `sepset` and `cluster_to` are canonical beliefs, and `cluster_to` is a generalized
belief

## Degeneracy

Propagating a belief requires the precision of the `cluster_from` belief (i.e. J if
`cluster_from` is canonical, or QΛQᵀ if it is generalized) to have a non-degenerate
submatrix J_I for the variables to be integrated out.
Problems arise if J_I has one or more 0 eigenvalues, or infinite values
(see [`marginalize`](@ref)).
If so, a [`BPPosDefException`](@ref) is returned **but not thrown**.
Downstream functions should try & catch these failures, and decide how to proceed.
See [`regularizebeliefs_bycluster!`](@ref) to reduce the prevalence of degeneracies.

## Output

- `nothing` if the message was calculated with success
- a [`BPPosDefException`](@ref) object if marginalization failed. In this case, *the error
  is not thrown*: downstream functions should check for failure (and may choose to throw the
  output error object).

## Warnings

- the `μ` parameter is not updated
- Does not check that `cluster_from` and `cluster_to` are of cluster type, or that `sepset`
  is of sepset type, but does check that the labels and scope of `sepset` are included in
  each cluster.
"""
function propagate_belief!(
    cluster_to::CanonicalBelief,
    sepset::CanonicalBelief,
    cluster_from::CanonicalBelief,
    residual::AbstractResidual,
)
    # 1. compute message: marginalize cluster_from to variables in sepset
    #    requires cluster_from.J[I,I] to be invertible, I = indices other than `keepind`
    #    marginalize sends BPPosDefException otherwise.
    # `keepind` can be empty (e.g. if `cluster_from` is entirely "clamped")
    keepind = scopeindex(sepset, cluster_from)
    h,J,g = try marginalize(cluster_from, keepind)
    catch ex
        isa(ex, BPPosDefException) && return ex # output the exception: not thrown
        rethrow(ex) # exception thrown if other than BPPosDefException
    end
    # calculate residual
    residual.Δh .= h .- sepset.h
    residual.ΔJ .= J .- sepset.J
    # 2. extend message to scope of cluster_to and propagate
    upind = scopeindex(sepset, cluster_to) # indices to be updated
    cluster_to.h[upind] .+= residual.Δh
    cluster_to.J[upind, upind] .+= residual.ΔJ
    cluster_to.g[1] += g - sepset.g[1]
    # 3. update sepset belief
    sepset.h[:] = h
    sepset.J[:] = J
    sepset.g[1] = g
    return nothing
end
function propagate_belief!(
    cluster_to::AbstractBelief,
    sepset::AbstractBelief,
    cluster_from::AbstractBelief,
    residual::AbstractResidual,
)
    propagate_belief!(cluster_to::AbstractBelief, sepset::AbstractBelief,
        cluster_from::AbstractBelief, scopeindex(sepset, cluster_from))
end
function propagate_belief!(
    cluster_to::GeneralizedBelief,
    sepset::GeneralizedBelief,
    cluster_from::GeneralizedBelief,
    keepind::Vector{T},
) where T <: Integer
    marginalize!(cluster_from, keepind)
    divide!(sepset, cluster_from)
    mult!(cluster_to, sepset) # handles scope extension and matching
    return nothing
end
function propagate_belief!(
    cluster_to::GeneralizedBelief,
    sepset::CanonicalBelief,
    cluster_from::CanonicalBelief,
    keepind::Vector{T},
) where T <: Integer
    h, J, g = marginalize(cluster_from, keepind)
    Δh, ΔJ, Δg = divide!(sepset, h, J, g)
    mult!(cluster_to, scopeindex(sepset, cluster_to), Δh, ΔJ, Δg)
    return nothing
end
function propagate_belief!(
    cluster_to::CanonicalBelief,
    sepset::CanonicalBelief,
    cluster_from::GeneralizedBelief,
    keepind::Vector{T}
) where T <: Integer
    marginalize!(cluster_from, keepind)
    Δh, ΔJ, Δg = divide!(sepset, cluster_from)
    upind = scopeindex(sepset, cluster_to)
    cluster_to.h[upind] .+= Δh
    cluster_to.J[upind,upind] .+= ΔJ
    cluster_to.g[1] += Δg
    return nothing
end