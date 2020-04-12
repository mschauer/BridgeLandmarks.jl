
"""
    GuidedProposal!{T,Ttarget,Taux,TL,Txobs0,TxobsT,Tnshapes,TmT,F} <: ContinuousTimeProcess{T}

struct that contains target, auxiliary process for each shape, time grid, observation at time 0, observations
at time T, number of shapes, and momenta in final state used for constructing the auxiliary processes
`guidrec` is a vector of GuidRecursions, which contains the results from the backward recursions and gpupdate step at time zero
"""
mutable struct GuidedProposal!{T,Ttarget,Taux,Txobs0,TelxobsT,TL,TmT,F} <: ContinuousTimeProcess{T}
    target::Ttarget                 # target diffusion P
    aux::Vector{Taux}               # auxiliary diffusion for each shape (Ptilde for each shape)
    tt::Vector{Float64}             # grid of time points on single segment (S,T]
    xobs0::Txobs0
    xobsT::Vector{TelxobsT}
    guidrec::Vector{TL}             # guided recursions on grid tt
    nshapes::Int64
    mT::TmT                         # momenta of final state used for defining auxiliary process
    endpoint::F

    function GuidedProposal!(target, aux, tt_,xobs0,xobsT, guidrec,nshapes,mT, endpoint=Bridge.endpoint)
        tt = collect(tt_)
        new{Bridge.valtype(target),typeof(target),eltype(aux),typeof(xobs0), eltype(xobsT),eltype(guidrec),typeof(mT),typeof(endpoint)}(
                    target, aux, tt,xobs0,xobsT, guidrec, nshapes, mT,endpoint)
    end
end

"""
    _b!((i,t), x::State, out::State, Q::GuidedProposal!,k)

Evaluate drift bᵒ of guided proposal at (t,x), write into out
"""
function _b!((i,t), x::State, out::State, Q::GuidedProposal!,k)
    Bridge.b!(t, x, out, Q.target)
    out .+= amul(t,x,Q.guidrec[k].Lt[i]' *
        (Q.guidrec[k].Mt[i] *(Q.xobsT[k]-Q.guidrec[k].μt[i]-Q.guidrec[k].Lt[i]*vec(x))),Q.target)
    out
end

"""
    σ!(t, x, dw, out, Q::GuidedProposal!)

Evaluate σ(t,x) dw and write into out
"""
σ!(t, x, dw, out, Q::GuidedProposal!) = σ!(t, x, dw, out, Q.target)

"""
    _r!((i,t), x::State, out::State, Q::GuidedProposal!,k)

Evaluate tilde_r (appearing in guiding term of guided proposal) at (t,x), write into out
"""
function _r!((i,t), x::State, out::State, Q::GuidedProposal!,k)
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
    gp!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposal!,k; skip = 0, ll0 = true)

Simulate guided proposal as specified in `Q` and compute loglikelihood for one shape,
starting from `x0`, using Wiener increments `W`

Returns logliklihood.
"""
function gp!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposal!,k; skip = 0, ll0 = true)
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
    gp!(::LeftRule,  X, x0, W, Q::GuidedProposal!; skip = 0, ll0 = true)

Simulate guided proposal and compute loglikelihood (vector version, multiple shapes)
"""
function gp!(::LeftRule,  X::Vector, x0, W, Q::GuidedProposal!; skip = 0, ll0 = true)
    soms  = zeros(deepeltype(x0), Q.nshapes)
    for k in 1:Q.nshapes
        soms[k] = gp!(LeftRule(), X[k],x0,W[k],Q,k ;skip=skip,ll0=ll0)
    end
    soms
end

"""
    getpars(Q::GuidedProposal!)

Extract parameters from GuidedProposal! `Q`, that is, `(a,c,γ)``
"""
function getpars(Q::GuidedProposal!)
    P = Q.target
    [P.a, P.c, getγ(P)]
end

"""
    putpars!(Q::GuidedProposal!,(aᵒ,cᵒ,γᵒ))

Update parameter values in GuidedProposal! `Q`, i.e. new values are written into `Q.target` and `Q.aux`
"""
function putpars!(Q::GuidedProposal!,(aᵒ,cᵒ,γᵒ))
    if isa(Q.target,MarslandShardlow)
        Q.target = MarslandShardlow(aᵒ,cᵒ,γᵒ,Q.target.λ, Q.target.n)
    elseif isa(Q.target,Landmarks)
        nfs = construct_nfs(Q.target.db, Q.target.nfstd, γᵒ)
        Q.target = Landmarks(aᵒ,cᵒ,Q.target.n,Q.target.db,Q.target.nfstd,nfs)
    end
    Q.aux = [auxiliary(Q.target,State(Q.xobsT[k],Q.mT)) for k in 1:Q.nshapes]
end

"""
    target(Q::GuidedProposal!) = Q.target
"""
    target(Q::GuidedProposal!) = Q.target

"""
    auxiliary(Q::GuidedProposal!,k) = Q.aux[k]

Extract auxiliary process of k-th shape.
"""
auxiliary(Q::GuidedProposal!,k) = Q.aux[k] # auxiliary process of k-th shape

"""
    constdiff(Q::GuidedProposal!)

If true, both the target and auxiliary process have constant diffusion coefficient.
"""
constdiff(Q::GuidedProposal!) = constdiff(target(Q)) && constdiff(auxiliary(Q,1))

"""
    update_Paux_xT!(Q, mTvec, obs_info)

Update State vector of auxiliary process for each shape.
For the k-th shape, the momentum gets replaced with `mTvec[k]`
"""
function update_Paux_xT!(Q, mTvec, obs_info)
    for k in Q.nshapes
        Q.aux[k] = auxiliary(Q.target,State(Q.xobsT[k],mTvec[k]))  # auxiliary process for each shape
    end
    update_guidrec!(Q, obs_info)
end
