"""
    struct that contains target, auxiliary process for each shape, time grid, observation at time 0, observations
        at time T, number of shapes, and momenta in final state used for constructing the auxiliary processes
    guidrec is a vector of GuidRecursions, which contains the results from the backward recursions and gpupdate step at time zero
"""
mutable struct GuidedProposal!{T,Ttarget,Taux,TL,Txobs0,TxobsT,Tnshapes,TmT,F} <: ContinuousTimeProcess{T}
    target::Ttarget                 # target diffusion P
    aux::Vector{Taux}               # auxiliary diffusion for each shape (Ptilde for each shape)
    tt::Vector{Float64}             # grid of time points on single segment (S,T]
    guidrec::Vector{TL}             # guided recursions on grid tt
    xobs0::Txobs0                   # observation at time 0
    xobsT::Vector{TxobsT}           # observations for each shape at time T
    nshapes::Int64                  # number of shapes
    mT::TmT                         # momenta of final state used for defining auxiliary process
    endpoint::F

    function GuidedProposal!(target, aux, tt_, guidrec, xobs0, xobsT, nshapes, mT, endpoint=Bridge.endpoint)
        tt = collect(tt_)
        new{Bridge.valtype(target),typeof(target),eltype(aux),eltype(guidrec),typeof(xobs0),eltype(xobsT),Int64,typeof(mT),typeof(endpoint)}(target, aux, tt, guidrec, xobs0, xobsT, nshapes, mT,endpoint)
    end
end

"""
    GuidRecursions defines a struct that contains all info required for computing the guiding term and
    likelihood (including ptilde term) for a single shape

Suppose t is the specified (fixed) time grid. Then the elements of the struct are:
Lt:     matrices L
Mt⁺:    matrices M⁺ (inverses of M)
M:      matrices M
μ:      vectors μ
Ht:     matrices H, where H = L' M L
Lt0:    L(0) (so obtained from L(0+) after gpupdate step incorporating observation xobs0)
Mt⁺0:   M⁺(0) (so obtained from M⁺(0+) after gpupdate step incorporating observation xobs0)
μt0:    μ(0) (so obtained μ(0+) after gpupdate step incorporating observation xobs0)
"""
mutable struct GuidRecursions{TL,TM⁺,TM, Tμ, TH, TLt0, TMt⁺0, Tμt0}
    Lt::Vector{TL}          # Lt on grid tt
    Mt⁺::Vector{TM⁺}        # Mt⁺ on grid tt
    Mt::Vector{TM}          # Mt on grid tt
    μt::Vector{Tμ}          # μt on grid tt
    Ht::Vector{TH}          # Ht on grid tt
    Lt0::TLt0               # Lt at time 0, after gpupdate step incorporating observation xobs0
    Mt⁺0::TMt⁺0             # inv(Mt) at time 0, after gpupdate step incorporating observation xobs0
    μt0::Tμt0               # μt at time 0, after gpupdate step incorporating observation xobs0

    function GuidRecursions(Lt, Mt⁺,Mt, μt, Ht, Lt0, Mt⁺0, μt0)
            new{eltype(Lt), eltype(Mt⁺), eltype(Mt),eltype(μt),eltype(Ht), typeof(Lt0), typeof(Mt⁺0), typeof(μt0)}(Lt, Mt⁺,Mt, μt, Ht, Lt0, Mt⁺0, μt0)
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

Simulate guided proposal and compute loglikelihood for one shape
Solve sde inplace and return loglikelihood (thereby avoiding 'double' computations)
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
function gp!(::LeftRule,  X, x0, W, Q::GuidedProposal!; skip = 0, ll0 = true)
    soms  = zeros(deepeltype(x0), Q.nshapes)
    for k in 1:Q.nshapes
        soms[k] = gp!(LeftRule(), X[k],x0,W[k],Q,k ;skip=skip,ll0=ll0)
    end
    soms
end

"""
    getpars(Q::GuidedProposal!)

Extract parameters from GuidedProposal! Q, that is, (a,c,γ)
"""
function getpars(Q::GuidedProposal!)
    P = Q.target
    [P.a, P.c, getγ(P)]
end

"""
    putpars!(Q::GuidedProposal!,(aᵒ,cᵒ,γᵒ))

Update parameter values in GuidedProposal! Q, i.e. new values are written into Q.target and Q.aux
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
