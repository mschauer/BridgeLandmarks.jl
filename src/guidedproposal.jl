
"""
    GuidedProposal{T,Ttarget,Taux,TL,Txobs0,TxobsT,Tnshapes,TmT,F} <: ContinuousTimeProcess{T}

struct that contains target, auxiliary process for each shape, time grid, observation at time 0, observations
at time T, number of shapes, and momenta in final state used for constructing the auxiliary processes
`guidrec` is a vector of GuidRecursions, which contains the results from the backward recursions and gpupdate step at time zero

## Arguments
```julia
target::Ttarget                 # target diffusion P
aux::Vector{Taux}               # auxiliary diffusion for each shape (Ptilde for each shape)
tt::Vector{Float64}             # grid of time points on single segment (S,T]
xobs0::Txobs0                   # observation at time 0
xobsT::Vector{TelxobsT}         # vector of observations at time T
guidrec::Vector{TL}             # guided recursions on grid tt
nshapes::Int64                  # number of shapes
mT::Vector{TmT}                 # vector of artificial momenta used for constructing auxiliary process (one for each shape)
endpoint::F
```

## Example (for landmarksmatching)
```julia
    n = 5
    P = MarslandShardlow(1.0, 2.0, 0.5, 0.0, n)
    xT = State(rand(PointF,n),rand(PointF,n))
    Paux = [BridgeLandmarks.auxiliary(P,xT)]
    tt = collect(0:0.05:1.0)
    xobs0 = rand(PointF,n)
    xobsT = [rand(PointF,n)]
    σobs = 0.01
    Σobs = fill([σobs^2 * one(UncF) for i in 1:n],2)
    obsinfo = BridgeLandmarks.set_obsinfo(xobs0,xobsT,Σobs, true,false)
    gr = [BridgeLandmarks.init_guidrec(tt,obsinfo)]
    nshapes = 1
    mT = [rand(PointF,n)]
    Q = BridgeLandmarks.GuidedProposal(P,Paux,tt,xobs0,xobsT,gr,nshapes,mT)
```
"""
struct GuidedProposal{T,Ttarget,Taux,Txobs0,TelxobsT,TL,TmT,F} <: ContinuousTimeProcess{T}
    target::Ttarget                 # target diffusion P
    aux::Vector{Taux}               # auxiliary diffusion for each shape (Ptilde for each shape)
    tt::Vector{Float64}             # grid of time points on single segment (S,T]
    xobs0::Txobs0                   # observation at time 0
    xobsT::Vector{TelxobsT}         # vector of observations at time T
    guidrec::Vector{TL}             # guided recursions on grid tt
    nshapes::Int64                  # number of shapes
    mT::Vector{TmT}                 # vector of artificial momenta used for constructing auxiliary process (one for each shape)
    endpoint::F

    function GuidedProposal(target, aux, tt_,xobs0,xobsT, guidrec,nshapes,mT, endpoint=Bridge.endpoint)
        tt = collect(tt_)
        new{Bridge.valtype(target),typeof(target),eltype(aux),typeof(xobs0), eltype(xobsT),eltype(guidrec),eltype(mT),typeof(endpoint)}(
                    target, aux, tt,xobs0,xobsT, guidrec, nshapes,mT, endpoint)
    end
end

"""
    set_guidrec(Q::GuidedProposal, gr) = GuidedProposal(Q.target, Q.aux, X.tt, Q.xobs, Q.xobsT, gr,Q.nshapes, Q.endpoint)

Update `guidrec` field of Q to `gr`
"""
set_guidrec(Q::GuidedProposal, gr) = GuidedProposal(Q.target, Q.aux, Q.tt, Q.xobs0, Q.xobsT, gr,Q.nshapes,Q.mT, Q.endpoint)

"""
    update_mT(Q, mTv, obsinfo)

Update State vector of auxiliary process for each shape.
For the k-th shape, the momentum gets replaced with `mTv[k]`

## Example
See documentation for GuidedProposal to contruct an example instance, say `Q`

```julia
    mTv = [zeros(PointF,5)]

    BridgeLandmarks.update_mT!(Q, mTv, obsinfo)
"""
function update_mT!(Q, mTv, obsinfo)
    for k in Q.nshapes
        Q.aux[k] = auxiliary(Q.target,State(Q.xobsT[k],mTv[k]))  # auxiliary process for each shape
    end
    update_guidrec!(Q, obsinfo)
end

"""
    construct_gp_xobsT(Q, xobsTᵒ)

Update xobsTᵒ into auxiliary process of Q, following by recomputing the backwards ODEs
"""
function construct_gp_xobsT(Q, xobsTᵒ)
    aux = [auxiliary(Q.target,State(xobsTᵒ[k],Q.mT[k])) for k in 1:Q.nshapes]
    GuidedProposal(Q.target, aux, Q.tt, Q.xobs0, xobsTᵒ, Q.guidrec, Q.nshapes, Q.mT)
end



"""
    _b!((i,t), x::State, out::State, Q::GuidedProposal,k)

Evaluate drift bᵒ of guided proposal at (t,x), write into out
"""
function _b!((i,t), x::State, out::State, Q::GuidedProposal,k)
    Bridge.b!(t, x, out, Q.target)
    out .+= amul(t,x,Q.guidrec[k].Lt[i]' *
        (Q.guidrec[k].Mt[i] *(Q.xobsT[k]-Q.guidrec[k].μt[i]-Q.guidrec[k].Lt[i]*vec(x))),Q.target)
    out
end

"""
    σ!(t, x, dw, out, Q::GuidedProposal)

Evaluate σ(t,x) dw and write into out
"""
σ!(t, x, dw, out, Q::GuidedProposal) = σ!(t, x, dw, out, Q.target)

"""
    _r!((i,t), x::State, out::State, Q::GuidedProposal,k)

Evaluate tilde_r (appearing in guiding term of guided proposal) at (t,x), write into out
"""
function _r!((i,t), x::State, out::State, Q::GuidedProposal,k)
    out .= vecofpoints2state(Q.guidrec[k].Lt[i]' * (Q.guidrec[k].Mt[i] *(Q.xobsT[k]-Q.guidrec[k].μt[i]-Q.guidrec[k].Lt[i]*vec(x))))
    out
end

"""
    lρtilde(x0, Q,k)

Compute log ρ̃(0,x_0,k), where k indexes shape
"""
function lρtilde(x0, Q,k)
  y = deepvec([Q.xobs0; Q.xobsT[k]] - Q.guidrec[k].μt0 - Q.guidrec[k].Lt0*vec(x0))
  M⁺0deep = deepmat(Q.guidrec[k].Mt⁺0)
  -0.5*logdet(M⁺0deep) -0.5*dot(y, M⁺0deep\y)
end

"""
    gp!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposal,k; skip = 0, ll0 = true)

Simulate guided proposal as specified in `Q` and compute loglikelihood for one shape,
starting from `x0`, using Wiener increments `W`

Returns logliklihood.
"""
function gp!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposal,k; skip = 0, ll0 = true)
    Pnt = eltype(x0)
    tt =  Xᵒ.tt
    Xᵒ.yy[1] .= deepvalue(x0)
    som::deepeltype(x0)  = 0.
    # initialise objects to write into srout and strout are vectors of Points
    dwiener = dimwiener(Q.target)
    srout = zeros(Pnt, dwiener)
    strout = zeros(Pnt, dwiener)
    x = copy(x0)
    rout = copy(x0)
    bout = copy(x0)
    btout = copy(x0)
    wout = copy(x0)
    if !constdiff(Q)
        At = Bridge.a((1,0), x0, auxiliary(Q,k))  # auxtimehomogeneous switch
        A = zeros(Unc{deepeltype(x0)}, 2Q.target.n,2Q.target.n)
    end
    for i in 1:length(tt)-1
        dt = tt[i+1]-tt[i]
        b!(tt[i], x, bout, target(Q)) # b(t,x)
        _r!((i,tt[i]), x, rout, Q,k) # tilder(t,x)
        σt!(tt[i], x, rout, srout, target(Q))      #  σ(t,x)' * tilder(t,x) for target(Q)
        Bridge.σ!(tt[i], x, srout*dt + W.yy[i+1] - W.yy[i], wout, target(Q)) # σ(t,x) (σ(t,x)' * tilder(t,x) + dW(t))
        # likelihood terms
        if i<=length(tt)-1-skip
            _b!((i,tt[i]), x, btout, auxiliary(Q,k))
            som += dot(bout-btout, rout) * dt
            if !constdiff(Q)
                σt!(tt[i], x, rout, strout, auxiliary(Q,k))  #  tildeσ(t,x)' * tilder(t,x) for auxiliary(Q)
                som += 0.5*Bridge.inner(srout) * dt    # |σ(t,x)' * tilder(t,x)|^2
                som -= 0.5*Bridge.inner(strout) * dt   # |tildeσ(t,x)' * tilder(t,x)|^2
                Bridge.a!((i,tt[i]), x, A, target(Q))
                som += 0.5*(dot(At,Q.guidrec[k].Ht[i]) - dot(A,Q.guidrec[k].Ht[i])) * dt
            end
        end
        x .= x + dt * bout + wout
        Xᵒ.yy[i+1] .= deepvalue(x)
    end
    if ll0
        logρ0 = lρtilde(x0,Q,k)
    else
        logρ0 = 0.0 # don't compute
    end
    copyto!(Xᵒ.yy[end], Bridge.endpoint(Xᵒ.yy[end],Q))
    som + logρ0
end

"""
    gp!(::LeftRule,  X, x0, W, Q::GuidedProposal; skip = 0, ll0 = true)

Simulate guided proposal and compute loglikelihood (vector version, multiple shapes)
"""
function gp!(::LeftRule,  X::Vector, x0, W, Q::GuidedProposal; skip = 0, ll0 = true)
    logliks  = zeros(deepeltype(x0), Q.nshapes)
    for k in 1:Q.nshapes
        logliks[k] = gp!(LeftRule(), X[k],x0,W[k],Q, k ;skip=skip,ll0=ll0)
    end
    logliks
end

function gp_pos!(::LeftRule,  X::Vector, q, p, W, Q::GuidedProposal; skip = 0, ll0 = true)
    logliks  = zeros(deepeltype(q), Q.nshapes)
    x0 = merge_state(q,p)
    for k in 1:Q.nshapes
        logliks[k] = gp!(LeftRule(), X[k],x0,W[k],Q, k ;skip=skip,ll0=ll0)
    end
    logliks
end

function gp_mom!(::LeftRule,  X::Vector, q, p, W, Q::GuidedProposal; skip = 0, ll0 = true)
    logliks  = zeros(deepeltype(p), Q.nshapes)
    x0 = merge_state(q,p)
    for k in 1:Q.nshapes
        logliks[k] = gp!(LeftRule(), X[k],x0,W[k],Q, k ;skip=skip,ll0=ll0)
    end
    logliks
end


"""
    getpars(Q::GuidedProposal)

Extract parameters from GuidedProposal `Q`, that is, `(a,c,γ)``
"""
function getpars(Q::GuidedProposal)
    P = Q.target
    [P.a, P.c, getγ(P)]
end

"""
    adjust_to_newpars(Q::GuidedProposal,(aᵒ,cᵒ,γᵒ),obsinfo)

Provide new parameter values for GuidedProposal `Q`, these are written into fields `target` and `aux`.
Returns a new instance of `GuidedProposal`, adjusted to the new set of parameters.
"""
function adjust_to_newpars(Q::GuidedProposal,θᵒ)
    (aᵒ,cᵒ,γᵒ) = θᵒ
    if isa(Q.target,MarslandShardlow)
        target = MarslandShardlow(aᵒ,cᵒ,γᵒ,Q.target.λ, Q.target.n)
    elseif isa(Q.target,Landmarks)
        nfs = construct_nfs(Q.target.db, Q.target.nfstd, γᵒ)
        target = Landmarks(aᵒ,cᵒ,Q.target.n,Q.target.db,Q.target.nfstd,nfs)
    end
    aux = [auxiliary(target,State(Q.xobsT[k],Q.mT[k])) for k in 1:Q.nshapes]
    GuidedProposal(target, aux, Q.tt, Q.xobs0, Q.xobsT, Q.guidrec,Q.nshapes, Q.mT)
end

"""
    target(Q::GuidedProposal) = Q.target
"""
    target(Q::GuidedProposal) = Q.target

"""
    auxiliary(Q::GuidedProposal,k::Int64) = Q.aux[k]

Extract auxiliary process of k-th shape.
"""
auxiliary(Q::GuidedProposal,k::Int64) = Q.aux[k] # auxiliary process of k-th shape

"""
    constdiff(Q::GuidedProposal)

If true, both the target and auxiliary process have constant diffusion coefficient.
"""
constdiff(Q::GuidedProposal) = constdiff(target(Q)) && constdiff(auxiliary(Q,1))
