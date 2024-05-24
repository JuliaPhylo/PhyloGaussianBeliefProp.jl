"""
    generalizedBelief{T<:Real,Vlabel<:AbstractVector,P<:AbstractMatrix{T},V<:AbstractVector{T},M} <: AbstractBelief{T}

A *generalized* belief is an exponential quadratic form, multiplied by a Dirac
measure:
    𝒟(x | Q,R,Λ,h,c,g) = exp(-(1/2)xᵀ(QΛQᵀ)x + (Qh)ᵀx + g)⋅δ(Rᵀx - c)

If x ∈ ℝᵐ (i.e. is m-dimensional), then [Q R] is m x m, such that Q and R are
orthogonal (i.e. QᵀR = 0).

Q is m x (m-k) and R is m x k, where 0 ≤ k ≤ m is the degrees of degeneracy.
h and c are respectively m x 1 and k x 1 vectors.

When k=0, a generalized belief reduces to the standard [`Belief`](@ref) (i.e. an
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
struct generalizedBelief{T<:Real,Vlabel<:AbstractVector,P<:AbstractMatrix{T},V<:AbstractVector{T},M} <: AbstractBelief{T}
    "Integer label for nodes in the cluster"
    nodelabel::Vlabel # StaticVector{N,Tlabel}
    "Total number of traits at each node"
    ntraits::Int
    "Matrix inscope[i,j] is `false` if trait `i` at node `j` is / will be
    removed from scope, to avoid issues from 0 precision or infinite variance; or
    when there is no data for trait `i` below node `j` (in which case tracking
    this variable is only good for prediction, not for learning parameters)."
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
    "metadata, e.g. index in cluster graph, of type (M) `Symbol` for clusters or
    Tuple{Symbol, Symbol} for edges"
    metadata::M
end

# todo: violates DRY! Remove later. Use traits instead.
nodelabels(b::generalizedBelief) = b.nodelabel
ntraits(b::generalizedBelief) = b.ntraits
inscope(b::generalizedBelief) = b.inscope
nodedimensions(b::generalizedBelief) = map(sum, eachslice(inscope(b), dims=2))
dimension(b::generalizedBelief)  = sum(inscope(b))
function Base.show(io::IO, b::generalizedBelief)
    disp = "generalized belief for " * (b.type == bclustertype ? "Cluster" : "SepSet") * " $(b.metadata),"
    disp *= " $(ntraits(b)) traits × $(length(nodelabels(b))) nodes, dimension $(dimension(b)).\n"
    disp *= "Node labels: "
    print(io, disp)
    print(io, nodelabels(b))
    print(io, "\ntrait × node matrix of non-degenerate beliefs:\n")
    show(io, inscope(b))
    # print(io, "\nexponential quadratic belief, parametrized by\nμ: $(b.μ)\nh: $(b.h)\nJ: $(b.J)\ng: $(b.g[1])\n")
end

"""
    generalizedBelief(b::Belief)

Constructor from a standard belief `b`.

Precision `b.J` is eigendecomposed into `Q*Λ*transpose(Q)`, where `Q` and `Λ`
are square with the same dimensions as `b.J`, and `Λ` is positive semidefinite.
"""
function generalizedBelief(b::Belief{T,Vlabel,P,V,M}) where {T,Vlabel,P,V,M}
    # J = SArray{Tuple{size(b.J)...}}(b.J) # `eigen` cannot be called on
    Q, Λ = LA.svd(b.J)
    m = size(b.J,1) # dimension
    k = MVector{1,Int64}(0) # 0 degrees of degeneracy
    R = MMatrix{m,m,T}(undef)
    c = MVector{m,T}(undef)
    generalizedBelief{T,Vlabel,P,V,M}(
        b.nodelabel,b.ntraits,b.inscope,
        b.μ,transpose(Q)*b.h,similar(transpose(Q)*b.h),Q,similar(Q),Λ,similar(Λ),b.g,similar(b.g),
        k,similar(k),R,similar(R),c,similar(c),
        b.type,b.metadata)
end

"""
    generalizedBelief(b::Belief, R::AbstractMatrix)

Constructor from a standard belief `b` and a constraint matrix `R`.

`R` is assumed to have the same number of rows as `b.R`, and all entries of
`b.c[1:size(R)[2]]` are assumed to be 0.
"""
function generalizedBelief(b::Belief{T,Vlabel,P,V,M}, R::AbstractMatrix{T}) where {T,Vlabel,P,V,M}
    gb = generalizedBelief(b)
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
function extend!(cluster_to::generalizedBelief, sepset::generalizedBelief)
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
    cluster_to.Qbuf[perm,:] .= view(cluster_to.Qbuf,:,:) # permute
    # copy sepset.Λbuf[1:(m2-k2)] to cluster_to.Λbuf
    cluster_to.Λbuf .= 0.0
    cluster_to.Λbuf[1:(m2-k2)] .= sepset.Λbuf[1:(m2-k2)]
    # extend sepset.Rbuf[:,1:k2] in cluster_to.Rbuf
    cluster_to.Rbuf[:,1:k2] .= 0.0
    cluster_to.Rbuf[1:length(upind),1:k2] .= view(sepset.Rbuf,:,1:k2)
    cluster_to.Rbuf[perm,:] .= view(cluster_to.Rbuf,:,:) # permute
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

Multiply two generalized beliefs.

One generalized belief (`sepset`) is multiplied into the other (`cluster_to`). The
parameters of `cluster_to` are updated by this product.

Note that:
- the incoming message is accessed from the buffer of `sepset`
- the scope of this incoming message is extended within the buffer of `cluster_to`
"""
function mult!(cluster_to::generalizedBelief, sepset::generalizedBelief)
    extend!(cluster_to, sepset) # extend scope of `sepset` within buffer of `cluster_to`
    # degrees of degeneracy
    k1 = cluster_to.k[1]
    k2 = cluster_to.kbuf[1] # extended sepset
    # contraints
    R = cluster_to.R
    R1 = R[:,1:k1]
    R2 = cluster_to.Rbuf[:,1:k2] # Rbuf
    c = cluster_to.c
    c1 = c[1:k1]
    c2 = cluster_to.cbuf[1:k2] # cbuf
    # precisions
    m1 = size(cluster_to.Q)[2]
    Q1 = cluster_to.Q[:,1:(m1-k1)]
    Λ1 = LA.Diagonal(cluster_to.Λ[1:(m1-k1)])
    m2 = size(cluster_to.Qbuf)[2] # extended sepset
    Q2 = cluster_to.Qbuf[:,1:(m2-k2)] # Qbuf
    Λ2 = LA.Diagonal(cluster_to.Λbuf[1:(m2-k2)]) # Λbuf
    # potentials
    h = cluster_to.h
    h1 = h[1:(m1-k1)]
    h2 = cluster_to.hbuf[1:(m2-k2)] # hbuf
    # project onto R2 onto colsp(Q1), then orthonormalize by qr
    V = LA.qr(Q1 * transpose(Q1) * R2)
    V = V.Q[:,findall(LA.diag(V.R) .!== 0.0)]
    Δk1 = size(V)[2] # increase in degrees of degeneracy
    b = (transpose(R2)*V) \ (c2 - transpose(R2)*R1*c1)
    # update c1
    c[(k1+1):(k1+Δk1)] .= b
    # update R1
    R[:,(k1+1):(k1+Δk1)] .= V
    # update degrees of degeneracy
    cluster_to.k[1] += Δk1
    k1 += Δk1
    # for updating h and g (using new R1 and c1)
    Q1tVb = transpose(Q1)*V*b
    Λ1Q1tVb = Λ1*Q1tVb
    Q2tR1c1 = transpose(Q2)*view(R,:,1:k1)*view(c,1:k1)
    Λ2Q2tR1c1 = Λ2*Q2tR1c1
    # update Λ1
    U = LA.nullspace(transpose(view(R,:,1:k1)))
    Z, Λ, _ = LA.svd(transpose(U)*(Q1*Λ1*transpose(Q1) + Q2*Λ2*transpose(Q2))*U)
    # update Q1
    cluster_to.Q[:,1:(m1-k1)] .= U*Z
    cluster_to.Λ[1:(m1-k1)] .= Λ
    # update h1
    h[1:(m1-k1)] .= transpose(view(cluster_to.Q,:,1:(m1-k1)))*(Q1*(h1-Λ1Q1tVb) +
        Q2*(h2-Λ2Q2tR1c1))
    # update g1
    cluster_to.g[1] += (cluster_to.gbuf[1] + transpose(h1-0.5*Λ1Q1tVb)*Q1tVb +
        transpose(h2-0.5*Λ2Q2tR1c1)*Q2tR1c1 - LA.logdet(transpose(R2)*V))[1]
    return
end

"""
    div!(sepset, cluster_from)

Divide two generalized beliefs.

One generalized belief (`cluster_from`) is divided by the other (`sepset`). The
parameters of `sepset` are updated by this division.

Note that:
- the numerator of this division is accessed from the buffer of `cluster_from`.
Its scope is assumed to be the same (this is not checked) as that of `sepset`.
- the quotient of this division is stored in the buffer of `sepset`
"""
function div!(sepset::generalizedBelief, cluster_from::generalizedBelief)
    # degrees of degeneracy
    k1 = cluster_from.kbuf[1]
    k2 = sepset.k[1]
    # contraints
    m = sum(sepset.inscope)
    R1 = cluster_from.Rbuf[1:m,1:k1]
    R2 = sepset.R[:,1:k2]
    c1 = cluster_from.cbuf[1:k1]
    # precisions
    Q1 = cluster_from.Qbuf[1:m,1:(m-k1)]
    Λ1 = LA.Diagonal(cluster_from.Λbuf[1:(m-k1)])
    Q2 = sepset.Q[:,1:(m-k2)]
    Λ2 = LA.Diagonal(sepset.Λ[1:(m-k2)])
    # potentials
    h1 = cluster_from.hbuf[1:(m-k1)]
    h2 = sepset.h[1:(m-k2)]

    ## compute quotient
    k = k1-k2 # degrees of degeneracy for quotient
    # save quotient.k to sepset.kbuf[1]
    sepset.kbuf[1] = k
    # save quotient.R to sepset.Rbuf[:,1:k]
    cluster_from.Q[:,(m-k1+1):(m+k)] .= R2 # use cluster_from.Q as a buffer for R2
    sepset.Rbuf[:,1:k] .= LA.nullspace(view(cluster_from.Q,:,1:(m+k)))
    # save quotient.c to sepset.cbuf[1:k]
    sepset.cbuf[1:k] = transpose(view(sepset.Rbuf,:,1:k))*R1*c1
    # save quotient.Λ to sepset.Λbuf[1:(m-k1)]
    Z, Λ, _ = LA.svd(Λ1-transpose(Q1)*Q2*Λ2*transpose(Q2)*Q1)
    sepset.Λbuf[:] .= 0.0 # reset buffer
    sepset.Λbuf[1:(m-k1)] .= Λ
    # save quotient.Q to sepset.Qbuf[:,1:(m-k1+k2)]
    sepset.Qbuf[:,1:(m-k1)] .= Q1*Z
    sepset.Qbuf[:,(m-k1+1):(m+k)] .= R2
    # save quotient.h to sepset.hbuf[1:(m1-k1+k2)]
    Q2tR1c1 = transpose(Q2)*R1*c1
    Λ2Q2tR1c1 = Λ2*Q2tR1c1
    sepset.hbuf[1:(m+k)] = (transpose(sepset.Qbuf[:,1:(m+k)]) *
        (Q1*h1 - Q2*(h2 - Λ2*transpose(Q2)*R1*c1)))
    # save quotient.g to sepset.gbuf[1]
    sepset.gbuf[1] = (@. cluster_from.gbuf[1] - sepset.g[1] .-
        transpose(h2-0.5*Λ2Q2tR1c1)*Q2tR1c1)[1]
    
    ## update sepset non-buffer fields
    sepset.k[1] = k1
    sepset.R[:,1:k1] = R1
    sepset.c[1:k1] = c1
    sepset.Q[:,1:(m-k1)] .= Q1
    sepset.Λ[1:(m-k1)] .= LA.diag(Λ1)
    sepset.h[1:(m-k1)] .= h1
    sepset.g[1] = cluster_from.gbuf[1]
    return
end

"""
    marg!(cluster_from, keep_index)

Marginalize a generalized belief (`cluster_from`) by integrating out all
variables at indices `keep_index`.

Note that:
- the resulting marginal / outgoing message is saved in the buffer of
`cluster_from`
"""
function marg!(cluster_from::generalizedBelief, keepind)
    m = length(keepind)
    m1 = size(cluster_from.Q)[1]
    k1 = cluster_from.k[1]
    Q1 = cluster_from.Q[keepind,1:(m1-k1)]
    R1 = cluster_from.R[keepind,1:k1]
    Λ1 = LA.Diagonal(view(cluster_from.Λ,1:(m1-k1)))
    U = LA.qr(Q1)
    k = sum(LA.diag(U.R) .== 0.0) # degrees of degeneracy
    cluster_from.kbuf[1] = k # update kbuf
    cluster_from.Rbuf[1:m,1:k] .= view(U.Q,:,findall(LA.diag(U.R) .== 0.0)) # update Rbuf
    cluster_from.cbuf[1:k] .= (transpose(view(cluster_from.Rbuf,1:m,1:k))*R1*
        view(cluster_from.c,1:k1)) # update cbuf
    U = U.Q[:,findall(LA.diag(U.R) .!== 0.0)] # columnspace(Q_x)
    V = LA.nullspace(U) # nullspace(Q_x)
    W = transpose(R1)*Q1
    if !isempty(W)
        W = LA.qr(W)
        W = W.Q[:,findall(LA.diag(W.R) .!== 0.0)] # columnspace(transpose(R_x)*Q_x)
    end
    Q2 = cluster_from.Q[setdiff(1:m1,keepind),1:(m1-k1)]
    R2 = cluster_from.R[setdiff(1:m1,keepind),1:k1]
    F = transpose(W*((R2*W)\Q2))
    G = (transpose(Q1)-F*transpose(R1))*U
    S = V * ((transpose(V)*Λ1*V) \ transpose(V))
    Z, Λ, _ = LA.svd(transpose(G)*(Λ1-Λ1*S*Λ1)*G)
    cluster_from.Λbuf[1:(m-k)] .= Λ
    cluster_from.Qbuf[1:m,1:(m-k)] .= U*Z
    Fc1 = F*cluster_from.c[1:k1]
    Λ1Fc1 = Λ1*Fc1
    h1_Λ1Fc1 = cluster_from.h[1:(m1-k1)] - Λ1Fc1
    cluster_from.hbuf[1:(m-k)] .= transpose(Z)*transpose(G)*(LA.I-Λ1*S)*(h1_Λ1Fc1)
    cluster_from.gbuf[1] = (@. cluster_from.g[1] +
        transpose(h1_Λ1Fc1+0.5*Λ1Fc1)*Fc1 + 0.5*transpose(h1_Λ1Fc1)*S*(h1_Λ1Fc1))[1]
    if !isempty(transpose(V)*Λ1*V)
        cluster_from.gbuf[1] -= 0.5*LA.logdet((1/2π)*transpose(V)*Λ1*V)[1]
    end
    if !isempty(R2*W)
        cluster_from.gbuf[1] -= 0.5*LA.logdet(transpose(R2*W)*R2*W)[1]
    end
    return
end