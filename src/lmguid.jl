import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

struct tuningpars_mcmc
    ρinit::Float64              # initial value for ρ
    maxnrpaths::Int64           # at most update maxnrpaths in the innovation-updating step
    δinit::Array{Float64,1}     # initial value for δ
    covθprop                    # RW-covariance on Normally distributed update on log(θ)
    η                           # stepsize cooling function
    adaptskip::Int64            # adapt mT (used in Paux) every adaptskip number of iterations
end



"""
    Observe V0 = L0 X0 + N(0,Σ0) and VT = LT X0 + N(0,ΣT)
    μT is just a vector of zeros (possibly remove later)
"""
struct ObsInfo{TLT,TΣT,TμT,TL0,TΣ0}
     LT::TLT
     ΣT::TΣT
     μT::TμT
     L0::TL0
     Σ0::TΣ0

    function ObsInfo(LT,ΣT,μT,L0,Σ0)
         new{typeof(LT),typeof(ΣT),typeof(μT),typeof(L0),typeof(Σ0)}(LT,ΣT,μT,L0,Σ0)
    end
end

"""
    Make ObsInfo object

    Three cases:
    1) obs_atzero=true: this refers to the case of observing one landmark configuration at times 0 and T
    2) obs_atzero=false & fixinitmomentato0=false: case of observing multiple shapes at time T,
        both positions and momenta at time zero assumed unknown
    3) obs_atzero=false & fixinitmomentato0=true: case of observing multiple shapes at time T,
        positions at time zero assumed unknown, momenta at time 0 are fixed to zero

    Note: the value for xobs0 passed to this function is only relevant in case obs_atzero=true
"""
function set_obsinfo(n, obs_atzero::Bool,fixinitmomentato0::Bool, Σobs,xobs0)
    Σobs0 = Σobs[1]; ΣobsT = Σobs[2]
    if obs_atzero
        L0 = LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n]  # pick position indices
        Σ0 = [(i==j) * Σobs0[i] for i in 1:n, j in 1:n]
        ΣT = [(i==j) * ΣobsT[i] for i in 1:n, j in 1:n]
    elseif !obs_atzero & !fixinitmomentato0
        L0 = Array{UncF}(undef,0,2*n)
        Σ0 = Array{UncF}(undef,0,0)
        xobs0 = Array{PointF}(undef,0)
        LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n]
        ΣT = [(i==j) * ΣobsT[i] for i in 1:n, j in 1:n]
    elseif !obs_atzero & fixinitmomentato0   # only update positions and fix initial state momenta to zero
        xobs0 = zeros(PointF,n)
        L0 = [((i+1)==j) * one(UncF) for i in 1:2:2n, j in 1:2n] # pick momenta indices
        LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n] # pick position indices
        Σ0 = [(i==j) * Σobs0[i] for i in 1:n, j in 1:n]
        ΣT = [(i==j) * ΣobsT[i] for i in 1:n, j in 1:n]
    end
    μT = zeros(PointF,n)
    xobs0, ObsInfo(LT,ΣT,μT,L0,Σ0)
end

"""
    GuidRecursions defines a struct that contains all info required for computing the guiding term and
    likelihood (including ptilde term) for a single shape
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
    Extract parameters from GuidedProposal! Q
"""
function getpars(Q::GuidedProposal!)
    P = Q.target
    [P.a, P.c, getγ(P)]
end


"""
    update parameter values in GuidedProposal! Q, i.e.
    new values are written into Q.target and Q.aux is updated accordingly
"""
function putpars!(Q::GuidedProposal!,(aᵒ,cᵒ,γᵒ))
    if isa(Q.target,MarslandShardlow)
        Q.target = MarslandShardlow(aᵒ,cᵒ,γᵒ,Q.target.λ, Q.target.n)
    elseif isa(Q.target,Landmarks)
        nfs = construct_nfs(Q.target.db, Q.target.nfstd, γᵒ) # need ot add db and nfstd to struct Landmarks
        Q.target = Landmarks(aᵒ,cᵒ,Q.target.n,Q.target.db,Q.target.nfstd,nfs)
    end
    Q.aux = [auxiliary(Q.target,State(Q.xobsT[k],Q.mT)) for k in 1:Q.nshapes]
end



"""
    Initialise (allocate memory) a struct of type GuidRecursions for a single shape
    guidres = init_guidrec((t,obs_info,xobs0)
"""
function init_guidrec(t,obs_info,xobs0)
    Pnt = eltype(obs_info.ΣT)
    Lt =  [copy(obs_info.LT) for s in t]
    Mt⁺ = [copy(obs_info.ΣT) for s in t]
    Mt = map(X -> InverseCholesky(lchol(X)),Mt⁺)
    μt = [copy(obs_info.μT) for s in t]
    H = obs_info.LT' * (obs_info.ΣT * obs_info.LT )
    Ht = [copy(H) for s in t]
    Lt0 = copy([obs_info.L0; obs_info.LT])

    m = size(obs_info.Σ0)[1]
    n = size(obs_info.ΣT)[2]
    if m==0
        Mt⁺0 = copy(obs_info.ΣT)
    else
        Mt⁺0 = [copy(obs_info.Σ0) zeros(Pnt,m,n); zeros(Pnt,n,m) copy(obs_info.ΣT)]
    end
    μt0 = [0*xobs0; copy(obs_info.μT)]
    GuidRecursions(Lt, Mt⁺, Mt, μt, Ht, Lt0, Mt⁺0, μt0)
end

"""
    Guided proposal update for newly incoming observation at time zero.
    Information on new observations at time zero is (L0, Σ0, xobs0)
    Values just after time zero, (Lt0₊, Mt⁺0₊, μt0₊) are updated to time zero, the result being
    written into (Lt0, Mt⁺0, μt0)
"""
function lm_gpupdate!(Lt0₊, Mt⁺0₊::Array{Pnt,2}, μt0₊, (L0, Σ0, xobs0), Lt0, Mt⁺0, μt0) where Pnt
    Lt0 .= [L0; Lt0₊]
    m = size(Σ0)[1]
    n = size(Mt⁺0₊)[2]
    if m==0
        Mt⁺0 .= Mt⁺0₊
    else
        Mt⁺0 .= [Σ0 zeros(Pnt,m,n); zeros(Pnt,n,m) Mt⁺0₊]
    end
    μt0 .= [0*xobs0; μt0₊]
end

"""
    Compute backward recursion for all shapes and write into field Q.guidrec
"""
function update_guidrec!(Q, obs_info)
    for k in 1:Q.nshapes  # for all shapes

        gr = Q.guidrec[k]
        # solve backward recursions;
        Lt0₊, Mt⁺0₊, μt0₊ =  guidingbackwards!(Lm(), Q.tt, (gr.Lt, gr.Mt⁺,gr.μt), Q.aux[k], obs_info)

        # perform gpupdate step at time zero
        lm_gpupdate!(Lt0₊, Mt⁺0₊, μt0₊, (obs_info.L0, obs_info.Σ0, Q.xobs0),gr.Lt0, gr.Mt⁺0, gr.μt0)
        # compute Cholesky decomposition of Mt at each time on the grid
        # need to symmetrize gr.Mt⁺; else AHS  gives numerical roundoff errors when mT \neq 0
        S = map(X -> 0.5*(X+X'), gr.Mt⁺)
        gr.Mt = map(X -> InverseCholesky(lchol(X)),S)

    #    gr.Mt = map(X -> InverseCholesky(lchol(X)),gr.Mt⁺) # old code
        # compute Ht at each time on the grid
        for i in 1:length(gr.Ht)
            gr.Ht[i] .= gr.Lt[i]' * (gr.Mt[i] * gr.Lt[i] )
        end
    end
end

"""
    Update momenta mT used for constructing the auxiliary process and recompute
    backward recursion
"""
function update_Paux_xT!(Q, mTvec, obs_info)
    for k in Q.nshapes
        Q.aux[k] = auxiliary(Q.target,State(Q.xobsT[k],mTvec[k]))  # auxiliary process for each shape
    end
    update_guidrec!(Q, obs_info)
end

struct Lm  end

"""
    Solve backwards recursions in L, M, μ parametrisation on grid t
    implicit: if true, Euler backwards is used for solving ODE for Lt, else Euler forwards is used

    Case lowrank=true still gives an error: fixme!
"""
function guidingbackwards!(::Lm, t, (Lt, Mt⁺, μt), Paux, obs_info; implicit=true, lowrank=false) #FIXME: add lowrank
    Mt⁺[end] .= obs_info.ΣT
    Lt[end] .= obs_info.LT
    μt[end] .= obs_info.μT

    B̃ = Bridge.B(0, Paux)          # does not depend on time
    β̃ = vec(Bridge.β(0,Paux))       # does not depend on time
    σ̃T = σ̃(0, Paux) # vanilla, no (possibly enclose with Matrix)
    dt = t[2] - t[1]
    oldtemp = (0.5*dt) * Bridge.outer(Lt[end] * σ̃T)
    if lowrank
        # TBA lowrank on σ̃T, and write into σ̃T
        error("not implemented")
    end

    for i in length(t)-1:-1:1
        dt = t[i+1]-t[i]
        if implicit
            Lt[i] .= Lt[i+1]/lu(I - dt* B̃, Val(false)) # should we use pivoting?
        else
            Lt[i] .=  Lt[i+1] * (I + B̃ * dt)
        end
        temp = (0.5*dt) * Bridge.outer(Lt[i] * σ̃T)
        Mt⁺[i] .= Mt⁺[i+1] + oldtemp + temp
        oldtemp = temp
        μt[i] .= μt[i+1] + 0.5 * (Lt[i] + Lt[i+1]) * β̃ * dt  # trapezoid rule
    end
    (Lt[1], Mt⁺[1], μt[1])
end

target(Q::GuidedProposal!) = Q.target
auxiliary(Q::GuidedProposal!,k) = Q.aux[k] # auxiliary process of k-th shape
constdiff(Q::GuidedProposal!) = constdiff(target(Q)) && constdiff(auxiliary(Q,1))

"""
    Evaluate drift bᵒ of guided proposal at (t,x), write into out
"""
function _b!((i,t), x::State, out::State, Q::GuidedProposal!,k)
    Bridge.b!(t, x, out, Q.target)
    out .+= amul(t,x,Q.guidrec[k].Lt[i]' * (Q.guidrec[k].Mt[i] *(Q.xobsT[k]-Q.guidrec[k].μt[i]-Q.guidrec[k].Lt[i]*vec(x))),Q.target)
    out
end

"""
    Evaluate σ(t,x) dW and write into out
"""
σ!(t, x, dw, out, Q::GuidedProposal!) = σ!(t, x, dw, out, Q.target)

"""
    Evaluate tilde_r (appearing in guiding term of guided proposal) at (t,x), write into out
"""
function _r!((i,t), x::State, out::State, Q::GuidedProposal!,k)
    out .= vecofpoints2state(Q.guidrec[k].Lt[i]' * (Q.guidrec[k].Mt[i] *(Q.xobsT[k]-Q.guidrec[k].μt[i]-Q.guidrec[k].Lt[i]*vec(x))))
    out
end

"""
    Compute log tildeρ(0,x_0,k) for the k-th shape
"""
function lρtilde(x0, Q,k)
  y = deepvec([Q.xobs0; Q.xobsT[k]] - Q.guidrec[k].μt0 - Q.guidrec[k].Lt0*vec(x0))
  M⁺0deep = deepmat(Q.guidrec[k].Mt⁺0)
  -0.5*logdet(M⁺0deep) -0.5*dot(y, M⁺0deep\y)
end

"""
    Simulate guided proposal and compute loglikelihood for one shape
    Solve sde inplace and return loglikelihood (thereby avoiding 'double' computations)
"""
function simguidedlm_llikelihood!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposal!,k; skip = 0, ll0 = true)
    Pnt = eltype(x0)
    tt =  Xᵒ.tt
    Xᵒ.yy[1] .= deepvalue(x0)
    som::deepeltype(x0)  = 0.
    #som = zero(deepeltype(x0))

    # initialise objects to write into
    # srout and strout are vectors of Points
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
    Simulate guided proposal and compute loglikelihood (vector version, multiple shapes)

    solve sde inplace and return loglikelihood (thereby avoiding 'double' computations)
"""
function simguidedlm_llikelihood!(::LeftRule,  X, x0, W, Q::GuidedProposal!; skip = 0, ll0 = true) # rather would like to dispatch on type and remove '_mv' from function name
    soms  = zeros(deepeltype(x0), Q.nshapes)
    for k in 1:Q.nshapes
        soms[k] = simguidedlm_llikelihood!(LeftRule(), X[k],x0,W[k],Q,k ;skip=skip,ll0=ll0)
    end
    soms
end

# convert dual to float, while retaining float if type is float
deepvalue(x::Float64) = x
deepvalue(x::ForwardDiff.Dual) = ForwardDiff.value(x)
deepvalue(x) = deepvalue.(x)

function deepvalue(x::State)
    State(deepvalue.(x.x))
end

"""
    update bridges for all shapes using Crank-Nicholsen scheme with parameter ρ (only in case the method is mcmc)
    Newly accepted bridges are written into (X,W), loglikelihood on each segment is written into vector ll

    update_path!(X,Xᵒ,W,Wᵒ,Wnew,ll,x, Q, ρ, accpcn)
"""
function update_path!(X,Xᵒ,W,Wᵒ,Wnew,ll,x, Q, ρ, accpcn, maxnrpaths)
#    nn = length(X[1].yy)
    x0 = deepvec2state(x)
    dw = dimwiener(Q.target)
    # From current state (x,W) with loglikelihood ll, update to (x, Wᵒ)
    for k in 1:Q.nshapes
        sample!(Wnew, Wiener{Vector{PointF}}())
        # for i in eachindex(W[1].yy) #i in nn
        #     Wᵒ.yy[i] .= ρ * W[k].yy[i] + sqrt(1-ρ^2) * Wnew.yy[i]
        # end

        # updateset: those indices for which the Wiener increment gets updated
        if dw <= maxnrpaths
            updateset = 1:dw
        else
            updateset = sample(1:dw, maxnrpaths, replace=false)
        end
        updatesetcompl = setdiff(1:dw, updateset)
        for i in eachindex(W[1].yy) #i in nn
            for j in updateset
                Wᵒ.yy[i][j] = ρ * W[k].yy[i][j] + sqrt(1-ρ^2) * Wnew.yy[i][j]
            end
            for j in updatesetcompl
                Wᵒ.yy[i][j] =  W[k].yy[i][j]
            end
        end


        llᵒ_ = simguidedlm_llikelihood!(LeftRule(), Xᵒ[k], x0, Wᵒ, Q,k;skip=sk)
        diff_ll = llᵒ_ - ll[k]
        if log(rand()) <= diff_ll
            for i in eachindex(W[1].yy)  #i in nn
                X[k].yy[i] .= Xᵒ[k].yy[i]
                W[k].yy[i] .= Wᵒ.yy[i]
            end
            #println("update innovation. diff_ll: ",round(diff_ll;digits=3),"  accepted")
            ll[k] = llᵒ_
            push!(accpcn, 1)
        else
            #println("update innovation. diff_ll: ",round(diff_ll;digits=3),"  rejected")
            push!(accpcn, 0)
        end
    end
    accpcn
end

"""
    Stochastic approximation for loglikelihood.

    Simulate guided proposal and compute loglikelihood for starting point x0,
    guided proposals defined by Q and Wiener increments in W.
    Guided proposals are written into X.
    Writes vector of loglikelihoods into llout.
    Returns sum of loglikelihoods
"""
function slogρ!(x0deepv, Q, W,X,priormom,llout) # stochastic approx to log(ρ)
    x0 = deepvec2state(x0deepv)
    lltemp = simguidedlm_llikelihood!(LeftRule(), X, x0, W, Q; skip=sk)   #overwrites X
    llout .= ForwardDiff.value.(lltemp)

    sum(lltemp) + logpdf(priormom, vcat(p(x0)...))
end
slogρ!(Q, W, X,priormom, llout) = (x) -> slogρ!(x, Q, W,X,priormom,llout)


"""
    update initial state
    X:  current iterate of vector of sample paths
    Xᵒ: vector of sample paths to write proposal into
    W:  current vector of Wiener increments
    ll: current value of loglikelihood
    x, xᵒ, ∇x, ∇xᵒ: allocated vectors for initial state and its gradient
    sampler: either sgd (not checked yet) or mcmc
    Q::GuidedProposal!
    δ: vector with MALA stepsize for initial state positions (δ[1]) and initial state momenta (δ[2])
    update:  can be :mala_pos, :mala_mom, :rmmala_pos
"""
function update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,
                sampler, Q::GuidedProposal!, δ, update,priormom)
    n = Q.target.n
    x0 = deepvec2state(x)
    P = Q.target

    llout = copy(ll)
    lloutᵒ = copy(ll)
    u = slogρ!(Q, W, X, priormom,llout)
    uᵒ = slogρ!(Q, W, Xᵒ, priormom,lloutᵒ)
    cfg = ForwardDiff.GradientConfig(u, x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal


    if sampler ==:sgd  # CHECK VALIDITY LATER
        mask = deepvec(State(0*x0.q, onemask(x0.p)))
        # StateW = PointF
        # sample!(W, Wiener{Vector{StateW}}())
        ForwardDiff.gradient!(∇x, u, x, cfg) # X gets overwritten but does not chang
        xᵒ = x .+ δ[2] * mask .* ∇x
        slogρ!(Q, W, X, priormom, llout)(xᵒ)

        obj = sum(llout)
        #obj = simguidedlm_llikelihood!(LeftRule(), X, deepvec2state(x), W, Q; skip=sk)
        accepted = 1
    end
    if sampler==:mcmc
        accinit = 0.0
        ll_incl0 = 0,0
        ll_incl0ᵒ = 0.0 # define because of scoping rules

        if update in [:mala_pos, :mala_mom]
            ForwardDiff.gradient!(∇x, u, x, cfg) # X gets overwritten but does not change
            ll_incl0 = sum(llout)
            if update==:mala_pos
                mask = deepvec(State(onemask(x0.q),  0*x0.p))
                stepsize = δ[1]
            elseif update==:mala_mom
                mask = deepvec(State(0*x0.q, onemask(x0.p)))
                stepsize = δ[2]
            end
            mask_id = (mask .> 0.1) # get indices that correspond to positions or momenta
            xᵒ .= x .+ .5 * stepsize * mask.* ∇x .+ sqrt(stepsize) .* mask .* randn(length(x))
            cfgᵒ = ForwardDiff.GradientConfig(uᵒ, xᵒ, ForwardDiff.Chunk{2*d*n}())                        # should be ".=" or just "="?
            ForwardDiff.gradient!(∇xᵒ, uᵒ, xᵒ, cfgᵒ)
            ll_incl0ᵒ = sum(lloutᵒ)
            ndistr = MvNormal(d * n,sqrt(stepsize))
            accinit = ll_incl0ᵒ - ll_incl0 -
                      logpdf(ndistr,(xᵒ - x - .5*stepsize .* mask.* ∇x)[mask_id]) +
                     logpdf(ndistr,(x - xᵒ - .5*stepsize .* mask.* ∇xᵒ)[mask_id])
        elseif update == :rmmala_pos
               ForwardDiff.gradient!(∇x, u, x, cfg) # X gets overwritten but does not change
               ll_incl0 = sum(llout)
               mask = deepvec(State(onemask(x0.q),  0*x0.p))
               stepsize = δ[1]
               mask_id = (mask .> 0.1) # get indices that correspond to positions or momenta
               dK = gramkernel(x0.q,P)
               ndistr = MvNormal(zeros(d*n),stepsize*dK)
               xᵒ = copy(x)
               xᵒ[mask_id] = x[mask_id] .+ .5 * stepsize * dK * ∇x[mask_id] .+ rand(ndistr)
               cfgᵒ = ForwardDiff.GradientConfig(uᵒ, xᵒ, ForwardDiff.Chunk{2*d*n}())
               ForwardDiff.gradient!(∇xᵒ, uᵒ, xᵒ, cfgᵒ)
               ll_incl0ᵒ = sum(lloutᵒ)

               x0ᵒ = deepvec2state(xᵒ)
               dKᵒ = gramkernel(x0ᵒ.q,P) #reshape([kernel(x0ᵒ.q[i]- x0ᵒ.q[j],P) * one(UncF) for i in 1:n for j in 1:n], n, n)
               ndistrᵒ = MvNormal(zeros(d*n),stepsize*dKᵒ)  #     ndistrᵒ = MvNormal(stepsize*deepmat(Kᵒ))

               accinit = ll_incl0ᵒ - ll_incl0 -
                         logpdf(ndistr,xᵒ[mask_id] - x[mask_id] - .5*stepsize * dK * ∇x[mask_id]) +
                        logpdf(ndistrᵒ,x[mask_id] - xᵒ[mask_id] - .5*stepsize * dKᵒ * ∇xᵒ[mask_id])
        elseif update == :rmmala_mom
             ForwardDiff.gradient!(∇x, u, x, cfg) # X gets overwritten but does not change
             ll_incl0 = sum(llout)
             mask = deepvec(State(0*x0.q, onemask(x0.p)))
             stepsize = δ[2]
             mask_id = (mask .> 0.1) # get indices that correspond to positions or momenta

             # proposal step
             dK = gramkernel(x0.q,P)
             inv_dK = inv(dK)
             ndistr = MvNormal(zeros(d*n),stepsize*inv_dK)
             xᵒ = copy(x)
             xᵒ[mask_id] = x[mask_id] .+ .5 * stepsize * inv_dK * ∇x[mask_id] .+  rand(ndistr)

             # reverse step
             cfgᵒ = ForwardDiff.GradientConfig(uᵒ, xᵒ, ForwardDiff.Chunk{2*d*n}())
             ForwardDiff.gradient!(∇xᵒ, uᵒ, xᵒ, cfgᵒ) # Xᵒ gets overwritten but does not change
             ll_incl0ᵒ = sum(lloutᵒ)
             ndistrᵒ = ndistr

             accinit = ll_incl0ᵒ - ll_incl0 -
                        logpdf(ndistr,xᵒ[mask_id] - x[mask_id] - .5*stepsize * inv_dK * ∇x[mask_id]) +
                       logpdf(ndistrᵒ,x[mask_id] - xᵒ[mask_id] - .5*stepsize * inv_dK * ∇xᵒ[mask_id])
       elseif update == :rmrw_mom
            u(x) # writes into llout
            ll_incl0 = sum(llout)
            mask = deepvec(State(0*x0.q, onemask(x0.p)))
            stepsize = δ[2]
            mask_id = (mask .> 0.1) # get indices that correspond to positions or momenta

            # proposal step
            dK = gramkernel(x0.q,P)
            inv_dK = inv(dK)
            ndistr = MvNormal(zeros(d*n),stepsize*inv_dK)
            xᵒ = copy(x)
            xᵒ[mask_id] = x[mask_id]  .+  rand(ndistr)
            # reverse step
            uᵒ(xᵒ)  # writes int lloutᵒ
            ll_incl0ᵒ = sum(lloutᵒ)
            accinit = ll_incl0ᵒ - ll_incl0
            # proposal is symmetric           logpdf(ndistr,xᵒ[mask_id]) + logpdf(ndistr,x[mask_id])
         end

        # MH acceptance decision
        if log(rand()) <= accinit
            #println("update initial state ", update, " accinit: ", round(accinit;digits=3), "  accepted")
            obj = ll_incl0ᵒ
            deepcopyto!(X, Xᵒ)
            x .= xᵒ
            ∇x .= ∇xᵒ
            ll .= lloutᵒ
            accepted = 1
        else
            #println("update initial state ", update, " accinit: ", round(accinit;digits=3), "  rejected")
            obj = ll_incl0
            accepted = 0
        end
    end
    obj, (kernel = update, acc = accepted)
end


"""
    For fixed Wiener increments and initial state, update parameters by random-walk-MH
"""
function update_pars!(obs_info,X, Xᵒ,W, Q, Qᵒ, x, ll, priorθ, covθprop)
    θ = getpars(Q)    #(P.a, P.c, getγ(P))
    distr = MvLogNormal(MvNormal(log.(θ),covθprop))
    θᵒ = rand(distr)
    distrᵒ = MvLogNormal(MvNormal(log.(θᵒ),covθprop))
    putpars!(Qᵒ,θᵒ)
    update_guidrec!(Qᵒ, obs_info)   # compute backwards recursion
    llᵒ = simguidedlm_llikelihood!(LeftRule(), Xᵒ, deepvec2state(x), W, Qᵒ; skip=sk)

    A = sum(llᵒ) - sum(ll) +
        logpdf(priorθ, θᵒ) - logpdf(priorθ, θ) +
        logpdf(distrᵒ, θ) - logpdf(distr, θᵒ)
    if log(rand()) <= A
        ll .= llᵒ
        deepcopyto!(Q.guidrec,Qᵒ.guidrec)
        Q.target = Qᵒ.target
        deepcopyto!(Q.aux,Qᵒ.aux)
        deepcopyto!(X,Xᵒ)
        accept = 1
    else
        accept = 0
    end
    (kernel = ":parameterupdate", acc = accept)
end

"""
    Adapt the step size for mala_mom updates.
    The adaptation is multiplication/divsion by exp(η(n))
"""
function adaptmalastep(n,accinfo,δ, η, update; adaptskip = 15, targetaccept=0.5)
    if mod(n,adaptskip)==0
        ind1 =  findall(first.(accinfo).== update)[end-adaptskip+1:end]
        recent_mean = mean(last.(accinfo)[ind1])
        if recent_mean > targetaccept
            δ *= exp(η(n))
        else
            δ *= exp(-η(n))
        end
    end
    δ
end

"""
    Adapt the step size for mala_mom updates.
    The adaptation is multiplication/divsion by exp(η(n))
"""
function adaptparstep(n,accinfo,covθprop, η;  adaptskip = 15, targetaccept=0.5)
    if mod(n,adaptskip)==0
        ind1 =  findall(first.(accinfo).==":parameterupdate")[end-adaptskip+1:end]
        recent_mean = mean(last.(accinfo)[ind1])
        if recent_mean > targetaccept
            covθprop *= exp(2*η(n))
        else
            covθprop *= exp(-2*η(n))
        end
    end
    covθprop
end

sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))
invsigmoid(z::Real) = log(z/(1-z))


"""
    Adapt the step size for mala_mom updates.
    The adaptation is multiplication/divsion by exp(η(n)) for log(ρ/(1-ρ))
"""

function adaptpcnstep(n, accpcn, ρ, nshapes, η; adaptskip = 15, targetaccept=0.5)
    if mod(n,adaptskip)==0
        recentvals = accpcn[end-adaptskip*nshapes+1:end]
        recentmean = mean(recentvals)
        if recentmean > targetaccept
            ρ = sigmoid(invsigmoid(ρ) - η(n))
        else
            ρ = sigmoid(invsigmoid(ρ) + η(n))
        end
    end
    ρ
end

################

"""
    Perform mcmc or sgd for landmarks model using the LM-parametrisation
    tt_:      time grid
    (xobs0,xobsT): observations at times 0 and T (at time T this is a vector)
    Σobs: array with covariance matrix of Gaussian noise assumed on each element of xobs0 and xobsT
    mT: vector of momenta at time T used for constructing guiding term
    P: target process

    obs_atzero: Boolean, if true there is an observation at time zero
    fixinitmomentato0: Boolean, if true we assume at time zero we observe zero momenta
    xinit: initial guess on starting state

    ITER: number of iterations
    subsamples: vector of indices of iterations that are to be saved
    tp: tuning pars for updates
    updatescheme: specify type of updates
    outdir: output directory for animation

    Returns:
    Xsave: saved iterations of all states at all times in tt_
    parsave: saved iterations of all parameter updates ,
    objvals: saved values of stochastic approximation to loglikelihood
    perc_acc: acceptance percentages (bridgepath - inital state)
"""
function lm_mcmc(t, (xobs0,xobsT), Σobs, mT, P,
          obs_atzero, fixinitmomentato0, ITER, subsamples,
          xinit, tp, priorθ, priormom, updatescheme,
          outdir)

    lt = length(t)
    StateW = PointF
    dwiener = dimwiener(P)
    # initialisation
    xobs0, obs_info = set_obsinfo(P.n,obs_atzero,fixinitmomentato0,Σobs,xobs0)
    nshapes = length(xobsT)
    guidrec = [init_guidrec(t,obs_info,xobs0) for k in 1:nshapes]  # memory allocation for backward recursion for each shape
    Paux = [auxiliary(P, State(xobsT[k],mT)) for k in 1:nshapes] # auxiliary process for each shape
    Q = GuidedProposal!(P,Paux,t,guidrec,xobs0,xobsT,nshapes,mT)
    update_guidrec!(Q, obs_info)   # compute backwards recursion

    X = [initSamplePath(t, xinit) for k in 1:nshapes]
    W = [initSamplePath(t,  zeros(StateW, dwiener)) for k in 1:nshapes]
    for k in 1:nshapes
        sample!(W[k], Wiener{Vector{StateW}}())
    end
    ll = simguidedlm_llikelihood!(LeftRule(), X, xinit, W, Q; skip=sk)

    # saving objects
    objvals = Float64[]             # keep track of (sgd approximation of the) loglikelihood
    Xsave = typeof(zeros(length(t) * P.n * 2 * d * nshapes))[]
    parsave = Vector{Float64}[]
    push!(Xsave, convert_samplepath(X))
    obj = sum(ll)
    push!(objvals, obj)
    push!(parsave, getpars(Q))

    # memory allocations
    Xᵒ = deepcopy(X)
    Qᵒ = deepcopy(Q)
    Wᵒ = initSamplePath(t,  zeros(StateW, dwiener))
    Wnew = initSamplePath(t,  zeros(StateW, dwiener))
    x = deepvec(xinit)
    xᵒ = deepcopy(x)
    ∇x = deepcopy(x)
    ∇xᵒ = deepcopy(x)

    accinfo = []                        # keeps track of accepted parameter and initial state updates
    accpcn = Int64[]                      # keeps track of nr of accepted pCN updates

    δ = deepcopy(tp.δinit)
    ρ = deepcopy(tp.ρinit)
    covθprop = deepcopy(tp.covθprop)



    for i in 1:ITER
        println();  println("iteration $i")

        for update in updatescheme
            if update == :innov
                #@timeit to "path update"
                accpcn = update_path!(X, Xᵒ, W, Wᵒ, Wnew, ll, x, Q, ρ, accpcn, tp.maxnrpaths)
                ρ = adaptpcnstep(i, accpcn, ρ, Q.nshapes, tp.η; adaptskip = tp.adaptskip)
            elseif update in [:mala_mom, :rmmala_mom, :rmrw_mom]
                #@timeit to "update mom"
                 obj, accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,:mcmc, Q, δ, update,priormom)
                push!(accinfo, accinfo_)
                δ[2] = adaptmalastep(i,accinfo,δ[2], tp.η, update; adaptskip = tp.adaptskip)
            elseif update in [:mala_pos, :rmmala_pos]
                #@timeit to "update pos"
                obj, accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,:mcmc, Q, δ, update,priormom)
                push!(accinfo, accinfo_)
                δ[1] = adaptmalastep(i,accinfo,δ[1], tp.η, update; adaptskip = tp.adaptskip)
            elseif update == :parameter
                #@timeit to "update par"
                accinfo_ = update_pars!(obs_info,X, Xᵒ,W, Q, Qᵒ, x, ll, priorθ, covθprop)
                push!(accinfo, accinfo_)
                covθprop = adaptparstep(i,accinfo,covθprop, tp.η;  adaptskip = tp.adaptskip)
            elseif update == :sgd
                obj, accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,:sgd, Q, δ, update,priormom)
            end
        end

        # adjust mT
        if mod(i,tp.adaptskip)==0 && i < 100
            mTvec = [X[k][lt][2].p  for k in 1:nshapes]     # extract momenta at time T for each shape
            update_Paux_xT!(Q, mTvec, obs_info)
        end

        # don't remove
        # update matching
        # direction, accinfo_ = update_matching(obs_info, X, Xᵒ,W, Q, Qᵒ, x, ll)
        # push!(accinfo, accinfo_)



        # save some of the results
        if i in subsamples
            # if mergepaths #transform two Xpaths
            #     Xrev = vcat([reverse(X[1].yy), X[2].yy[2:end]]...)
            #     ttrev = vcat([-reverse(t), t[2:end]]...)
            #     Xrevpath = [SamplePath(ttrev, Xrev)]
            #     push!(Xsave, convert_samplepath(Xrevpath))
            # else
                push!(Xsave, convert_samplepath(X))
            #end
        end
        push!(parsave, getpars(Q))
        push!(objvals, obj)


        println("ρ ", ρ ,",   δ ", δ, ",   covθprop ", covθprop)


        # if write_at_half # write state halfway of lastindex
        #     indexhalf = searchsortedfirst(t, t[end]/2.0)
        #     save("forwardhalfway.jld", "Xhalf",X[1].yy[indexhalf])
        # end
    end


    Xsave, parsave, objvals, accpcn, accinfo, δ, ρ, covθprop
end
