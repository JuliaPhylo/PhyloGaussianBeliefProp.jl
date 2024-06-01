"""
    GeneralizedBelief{T<:Real,Vlabel<:AbstractVector,P<:AbstractMatrix{T},V<:AbstractVector{T},M} <: AbstractBelief{T}

A *generalized* belief is an exponential quadratic form, multiplied by a Dirac
measure:
    𝒟(x | Q,R,Λ,h,c,g) = exp(-(1/2)xᵀ(QΛQᵀ)x + (Qh)ᵀx + g)⋅δ(Rᵀx - c)

If x ∈ ℝᵐ (i.e. is m-dimensional), then [Q R] is m x m, such that Q and R are
orthogonal (i.e. QᵀR = 0).

Q is m x (m-k) and R is m x k, where 0 ≤ k ≤ m is the degrees of degeneracy.
h and c are respectively m x 1 and k x 1 vectors.

When k=0, a generalized belief reduces to the standard [`CanonicalBelief`](@ref) (i.e. an
exponential quadratic form) with canonical parameters (J₁=QΛQᵀ, h₁=Qh, g₁=g).

Note that:
- the `Q`, `R` fields store m x m matrices, and the `h` and `c` fields store
m x 1 vectors. Q, R, h, c in the above equations are accessed as: `Q[:,1:(m-k)]`,
`R[:,1:k]`, `h[1:(m-k)]`, `c[1:k]`.
- the `Λ` field stores an m x 1 vector. Λ in the above equation is accessed as:
`Diagonal(Λ[1:(m-k)])`
- the `h` field refers to the generalized belief parameter h, **not**
the canonical parameter h₁ for a standard belief.
- if QΛQᵀ is positive-definite (i.e. `Λ` has no 0-entries), then μ = (QΛQᵀ)⁻¹Qh
= QΛ⁻¹h. This is a mean estimate for x if k=0, and for QQᵀx if k>0. The `μ` field
is only updated as needed.

`hbuf`, `Qbuf`, `Λbuf`, `gbuf`, `kbuf`, `Rbuf`, `cbuf` are matrices/vectors of
the same dimensions as `h`, `Q`, `Λ`, `g`, `k`, `R`, `c` and are meant to act as
buffers to reduce allocations during belief-update operations such as scope
extension (for incoming messages) and division (of an outgoing message by a
sepset belief).
"""
struct GeneralizedBelief{
    T<:Real,
    Vlabel<:AbstractVector,
    P<:AbstractMatrix{T},
    V<:AbstractVector{T},
    M,
} <: AbstractBelief{T}
    "Integer label for nodes in the cluster"
    nodelabel::Vlabel
    "Total number of traits at each node"
    ntraits::Int
    """
    Matrix inscope[i,j] is `false` if trait `i` at node `j` is / will be
    removed from scope, to avoid issues from 0 precision or infinite variance; or
    when there is no data for trait `i` below node `j` (in which case tracking
    this variable is only good for prediction, not for learning parameters).
    """
    inscope::BitArray
    μ::V
    h::V
    hbuf::V
    "eigenbasis for precision"
    Q::P
    Qbuf::P
    "eigenvalues of precision"
    Λ::V
    Λbuf::V
    g::MVector{1,T}
    gbuf::MVector{1,T}
    "degrees of degeneracy"
    k::MVector{1,Int64}
    kbuf::MVector{1,Int64}
    "constraint matrix: linear dependencies among inscope variables"
    R::P
    Rbuf::P
    "offset"
    c::V
    cbuf::V
    type::BeliefType
    """metadata, e.g. index in cluster graph, of type (M) `Symbol` for clusters or
    Tuple{Symbol, Symbol} for edges"""
    metadata::M
end

showname(::GeneralizedBelief) = "generalized belief"
function Base.show(io::IO, b::GeneralizedBelief)
    show_name_scope(io, b)
    # todo: show major fields of b
    # print(io, "\nexponential quadratic belief, parametrized by\nμ: $(b.μ)\nh: $(b.h)\nJ: $(b.J)\ng: $(b.g[1])\n")
end

"""
    GeneralizedBelief(b::CanonicalBelief)

Constructor from a canonical belief `b`.

Precision `b.J` is eigendecomposed into `Q Λ transpose(Q)`, where `Q` and `Λ`
are square with the same dimensions as `b.J`, and `Λ` is positive semidefinite.
"""
function GeneralizedBelief(b::CanonicalBelief{T,Vlabel,P,V,M}) where {T,Vlabel,P,V,M}
    # J = SArray{Tuple{size(b.J)...}}(b.J) # `eigen` cannot be called on
    Q, Λ = LA.svd(b.J)
    m = size(b.J,1) # dimension
    k = MVector{1,Int64}(0) # 0 degrees of degeneracy
    R = MMatrix{m,m,T}(undef)
    c = MVector{m,T}(undef)
    GeneralizedBelief{T,Vlabel,P,V,M}(
        b.nodelabel,b.ntraits,b.inscope,
        b.μ,transpose(Q)*b.h,similar(transpose(Q)*b.h),Q,similar(Q),Λ,similar(Λ),b.g,similar(b.g),
        k,similar(k),R,similar(R),c,similar(c),
        b.type,b.metadata)
end

"""
    GeneralizedBelief(b::CanonicalBelief, R::AbstractMatrix)

Constructor from a standard belief `b` and a constraint matrix `R`.

`R` is assumed to have the same number of rows as `b.R`, and all entries of
`b.c[1:size(R)[2]]` are assumed to be 0.
"""
function GeneralizedBelief(b::CanonicalBelief{T,Vlabel,P,V,M}, R::AbstractMatrix{T}) where {T,Vlabel,P,V,M}
    gb = GeneralizedBelief(b)
    m, k = size(R)
    gb.R[:,1:k] .= R # R should have the same no. of rows as gb.R
    gb.Q[:,1:(m-k)] .= LA.nullspace(transpose(R))
    gb.Λ[1:(m-k)] .= LA.diag(transpose(view(gb.Q,:,1:(m-k)))*b.J*view(gb.Q,:,1:(m-k)))
    gb.h[1:(m-k)] .= transpose(view(gb.Q,:,1:(m-k)))*b.h
    gb.k[1] = k
    gb.c[:,1:k] .= 0
    gb
end

"""
    extend!(cluster_to, sepset)

Extend the scope of one generalized belief (`sepset`) to the scope of another
(`cluster_to`) so that multiplication (see [`mult!`](@ref)) can be applied.
Note that:
- the incoming message is accessed from the buffer of `sepset`
- the scope of this incoming message is extended within the buffer of `cluster_to`
- the labels in `sepset` are assumed to be ordered as in `cluster_to`
"""
function extend!(cluster_to::GeneralizedBelief, sepset::GeneralizedBelief)
    # indices in `cluster_to` inscope variables of `sepset` inscope variables
    upind = scopeindex(sepset, cluster_to)
    m2 = size(sepset.Q)[1]
    k2 = sepset.kbuf[1]
    # copy sepset.kbuf[1] to cluster_to.kbuf
    cluster_to.kbuf[1] = k2
    #= extend sepset.Qbuf[:,1:(m2-k2)] in cluster_to.Qbuf
    Q is extended to P[Q 0; 0 I], where P is a permutation matrix (determined by
    upind) and P*[QΛQᵀ 0; 0 0]*Pᵀ = (P*[Q 0; 0 I])*[Λ 0; 0 0]*([Q 0; 0 I]ᵀ*Pᵀ) =#
    cluster_to.Qbuf .= 0.0 # reset buffer
    cluster_to.Qbuf[LA.diagind(cluster_to.Qbuf)] .= 1.0 # 1s along the diagonal
    cluster_to.Qbuf[1:length(upind),1:(m2-k2)] .= view(sepset.Qbuf,:,1:(m2-k2)) # store Q in buffer
    m1 = size(cluster_to.Q)[1]
    perm = [upind;setdiff(1:m1,upind)]
    cluster_to.Qbuf[perm,:] .= cluster_to.Qbuf[:,:] # permute
    # copy sepset.Λbuf[1:(m2-k2)] to cluster_to.Λbuf
    cluster_to.Λbuf .= 0.0
    cluster_to.Λbuf[1:(m2-k2)] .= sepset.Λbuf[1:(m2-k2)]
    # extend sepset.Rbuf[:,1:k2] in cluster_to.Rbuf
    cluster_to.Rbuf[:,1:k2] .= 0.0
    cluster_to.Rbuf[1:length(upind),1:k2] .= view(sepset.Rbuf,:,1:k2)
    cluster_to.Rbuf[perm,:] .= cluster_to.Rbuf[:,:] # permute
    # extend sepset.hbuf[1:(m2-k2)] in cluster_to.hbuf
    cluster_to.hbuf .= 0.0
    cluster_to.hbuf[1:(m2-k2)] .= view(sepset.hbuf,1:(m2-k2))
    # copy sepset.cbuf[1:k2] to cluster_to.cbuf
    cluster_to.cbuf[1:k2] .= view(sepset.cbuf,1:k2)
    # copy sepset.gbuf[1] to cluster_to.gbuf
    cluster_to.gbuf[1] = sepset.gbuf[1]
    return
end

"""
    mult!(cluster_to, sepset)

Multiply two generalized beliefs, accessed from `cluster_to` and the buffer of
`sepset`.

The parameters of `sepset`'s buffer (i.e. the incoming message) are extended
(and matched), within the buffer of `cluster_to`, to the scope of `cluster_to`'s
parameters before multiplication is carried out. The parameters of `cluster_to`
are updated to those of the product.
"""
function mult!(cluster_to::GeneralizedBelief, sepset::GeneralizedBelief)
    ## extend scope of incoming message and save parameters to cluster_to buffer
    extend!(cluster_to, sepset)

    # degrees of degeneracy
    k1 = cluster_to.k[1]
    k2 = cluster_to.kbuf[1]
    # contraints
    R = cluster_to.R
    R1 = R[:,1:k1]
    R2 = cluster_to.Rbuf[:,1:k2]
    c = cluster_to.c
    c1 = c[1:k1]
    c2 = cluster_to.cbuf[1:k2]
    # precisions
    m1 = size(cluster_to.Q)[1]
    Q1 = cluster_to.Q[:,1:(m1-k1)]
    Λ1 = cluster_to.Λ[1:(m1-k1)]
    m2 = size(cluster_to.Qbuf)[1]
    Q2 = cluster_to.Qbuf[:,1:(m2-k2)]
    Λ2 = cluster_to.Λbuf[1:(m2-k2)]
    # potentials
    h = cluster_to.h
    h1 = h[1:(m1-k1)]
    h2 = cluster_to.hbuf[1:(m2-k2)]

    ## compute product and save parameters to cluster_to
    # constraint
    V = LA.qr(Q1*transpose(Q1)*R2) # project R2 onto colsp(Q1)
    V = V.Q[:,findall(LA.diag(V.R) .!== 0.0)] # orthonormal basis for colsp(Q1*Q1ᵀ*R2)
    Δk1 = size(V)[2] # increase in degrees of degeneracy
    R[:,(k1+1):(k1+Δk1)] = V
    R2tV = transpose(R2)*V
    if k2 == Δk1 # transpose(R2)*V is square
        b = R2tV \ (c2 - transpose(R2)*R1*c1)
        lgdet = 2*LA.logdet(R2tV)
    else
        VtR2R2tV = transpose(R2tV)*R2tV
        b = (VtR2R2tV \ transpose(R2tV))*(c2 - transpose(R2)*R1*c1)
        lgdet = LA.logdet(VtR2R2tV)
    end
    c[(k1+1):(k1+Δk1)] = b
    # degrees of degeneracy
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
    cluster_to.g[1] += cluster_to.gbuf[1] + transpose(h1-0.5*Λ1Q1tVb)*Q1tVb +
        transpose(h2-0.5*Λ2Q2tR1c1)*Q2tR1c1 - 0.5*lgdet
    return
end

"""
    div!(sepset, cluster_from)

Divide two generalized beliefs. The dividend is accessed from the buffer of
`cluster_from`, and the divisor is accessed from `sepset`.

The parameters of `sepset` are updated to those of the dividend (i.e. the message
sent from `cluster_from`), and the parameters of `sepset`'s buffer are updated
to those of the quotient (i.e. [`MessageResidual`](@ref)) from this division.

Assumptions (not checked):
- Both dividend and divisor are assumed to have matching scopes
- cluster_from.kbuf[1] ≥  sepset.k[1] (i.e. the degrees of degeneracy for the
divisor cannot exceed that of the dividend) for division to be well-defined
"""
function div!(sepset::GeneralizedBelief, cluster_from::GeneralizedBelief)
    # degrees of degeneracy
    k1 = cluster_from.kbuf[1]
    k2 = sepset.k[1]
    # contraints
    m = size(sepset.Q)[1] # m ≥ max(k1,k2)
    R1 = cluster_from.Rbuf[1:m,1:k1]
    R2 = sepset.R[:,1:k2] # m x k2
    c1 = cluster_from.cbuf[1:k1]
    # precisions
    Q1 = cluster_from.Qbuf[1:m,1:(m-k1)]
    Λ1 = cluster_from.Λbuf[1:(m-k1)]
    Q2 = sepset.Q[:,1:(m-k2)]
    Λ2 = sepset.Λ[1:(m-k2)]
    # potentials
    h1 = cluster_from.hbuf[1:(m-k1)]
    h2 = sepset.h[1:(m-k2)]

    ## compute quotient and save parameters to sepset buffer
    # degrees of degeneracy
    k = k1-k2
    k ≥ 0 || error("quotient cannot have negative degrees of degeneracy")
    sepset.kbuf[1] = k
    # constraint
    cluster_from.Q[1:m,(m-k1+1):(m-k)] = R2 # use cluster_from.Q as a buffer for R2
    sepset.Rbuf[:,1:k] = LA.nullspace(view(cluster_from.Q,:,1:(m-k)))
    sepset.cbuf[1:k] = transpose(view(sepset.Rbuf,:,1:k))*R1*c1
    # precision
    Z, Λ = LA.svd(LA.Diagonal(Λ1)-transpose(Q1)*Q2*LA.Diagonal(Λ2)*transpose(Q2)*Q1)
    sepset.Λbuf .= 0.0 # reset buffer
    sepset.Λbuf[1:(m-k1)] = Λ
    sepset.Qbuf[:,1:(m-k1)] = Q1*Z
    sepset.Qbuf[:,(m-k1+1):(m-k)] = R2
    # potential
    Q2tR1c1 = transpose(Q2)*R1*c1
    Λ2Q2tR1c1 = Λ2.*Q2tR1c1
    sepset.hbuf[1:(m-k)] = (transpose(view(sepset.Qbuf,:,1:(m-k))) *
        (Q1*h1 - Q2*(h2 - Λ2Q2tR1c1)))
    # normalization constant
    sepset.gbuf[1] = cluster_from.gbuf[1] - sepset.g[1] -
        transpose(h2-0.5*Λ2Q2tR1c1)*Q2tR1c1
    
    ## update sepset (non-buffer) parameters to those of the dividend
    sepset.k[1] = k1
    sepset.R[:,1:k1] = R1
    sepset.c[1:k1] = c1
    sepset.Q[:,1:(m-k1)] = Q1
    sepset.Λ[1:(m-k1)] = Λ1
    sepset.h[1:(m-k1)] = h1
    sepset.g[1] = cluster_from.gbuf[1]
    return
end

"""
    marg!(cluster_from, keep_index)

Marginalize a generalized belief, accessed from `cluster_from`, by integrating
out all variables at indices `keep_index`.

The parameters of `cluster_from`'s buffer are updated to those of the marginal.
"""
function marg!(cluster_from::GeneralizedBelief, keepind)
    mm = length(keepind) # dimension for marginal
    m1 = size(cluster_from.Q)[1]
    k = cluster_from.k[1] # k ≤ m1
    Q1 = cluster_from.Q[keepind,1:(m1-k)] # mm x (m1-k)
    R1 = cluster_from.R[keepind,1:k] # mm x k
    Λ = LA.Diagonal(view(cluster_from.Λ,1:(m1-k))) # matrix, not vector
    
    ## compute marginal and save parameters to cluster_from buffer
    # degrees of degeneracy (todo: set threshold for a zero singular value)
    U1, S1, V1 = LA.svd(Q1; full=true) # U1: mm x mm, S1: min(mm,m1-k) x 1, V1: (m1-k) x (m1-k)
    nonzeroind = findall(S1 .!= 0.0) # indices for non-zero singular values
    zeroind = setdiff(1:(m1-k), nonzeroind) # indices for zero singular values
    km = mm - length(nonzeroind)
    cluster_from.kbuf[1] = km
    # constraint
    cluster_from.Rbuf[1:mm,1:km] = view(U1,:,setdiff(1:mm,nonzeroind)) # mm x km
    cluster_from.cbuf[1:km] = transpose(view(cluster_from.Rbuf,1:mm,1:km))*R1*
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
    cluster_from.Λbuf[1:(mm-km)] = Λm
    cluster_from.Qbuf[1:mm,1:(mm-km)] = view(U1,:,nonzeroind)*Z
    # potential
    Fc = F*cluster_from.c[1:k] # (m1-k) x 1
    ΛFc = Λ*Fc # (m1-k) x 1
    h_ΛFc = cluster_from.h[1:(m1-k)] - ΛFc
    cluster_from.hbuf[1:(mm-km)] = transpose(Z)*transpose(G)*(LA.I-Λ*S)*(h_ΛFc)
    # normalization constant
    cluster_from.gbuf[1] = cluster_from.g[1] + transpose(h_ΛFc+0.5*ΛFc)*Fc
    if m1 != k # Λ: (m1-k) x (m1-k) not empty
        # if Λ empty but V is not, then VᵀΛV defaults to a zero matrix with ∞ logdet!
        cluster_from.gbuf[1] -= 0.5*LA.logdet((1/2π)*transpose(V)*Λ*V)
    end
    if k > 0 # R2ᵀR2: k x k not empty
        cluster_from.gbuf[1] -= 0.5*LA.logdet(transpose(R2*W)*R2*W)
    end
    return
end

"""
    integratebelief!(b::GeneralizedBelief)

Return `(μ, g)`, where `μ` is the conditional mean of inscope nodes in `b`, and
`g` is the normalization constant from integrating out all inscope nodes in `b`.

Note that:
- `b` does not necessarily have k=0 degrees of degeneracy
- `μ` is the mean estimate for the inscope nodes x if k=0, and for QQᵀx if k>0
- it is assumed that QΛQᵀ is positive-definite (i.e. Λ has no 0-entries)
"""
function integratebelief!(b::GeneralizedBelief)
    m = size(b.Q)[1]
    k = b.k[1]
    μ = view(b.Q,:,1:(m-k))*(view(b.h,1:(m-k)) ./ view(b.Λ,1:m-k))
    messageg = b.g[1] + (m*log(2π) - log(prod(view(b.Λ,1:(m-k)))) +
        sum(view(b.h,1:(m-k)) .^2 ./ view(b.Λ,1:(m-k))))/2
    return (μ, messageg)
end

"""
    propagate_belief!(cluster_to, sepset, cluster_from, residual)

Update the parameters of the generalized beliefs `cluster_to` and `sepset`, by
marginalizing the generalized belief `cluster_from` to the scope of `sepset` and
passing that message.

Note that:
- the "residual" (i.e. change in `sepset`'s parameters) can be accessed from
`sepset`'s buffer
"""
function propagate_belief!(
    cluster_to::GeneralizedBelief,
    sepset::GeneralizedBelief,
    cluster_from::GeneralizedBelief
    )
    keepind = scopeindex(sepset, cluster_from)
    marg!(cluster_from, keepind)
    div!(sepset, cluster_from)
    mult!(cluster_to, sepset) # handles scope extension and matching
    return
end