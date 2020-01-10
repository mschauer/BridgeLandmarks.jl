import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

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
"""
function set_obsinfo(n, obs_atzero::Bool,fixinitmomentato0::Bool, Σobs,xobs0)
    if obs_atzero
        L0 = LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n]  # pick position indices
        Σ0 = ΣT = [(i==j) * Σobs[i] for i in 1:n, j in 1:n]
    elseif !obs_atzero & !fixinitmomentato0
        L0 = Array{UncF}(undef,0,2*n)
        Σ0 = Array{UncF}(undef,0,0)
        xobs0 = Array{PointF}(undef,0)
        LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n]
        ΣT = [(i==j) * Σobs[i] for i in 1:n, j in 1:n]
    elseif !obs_atzero & fixinitmomentato0   # only update positions and fix initial state momenta to zero
        xobs0 = zeros(PointF,n)
        L0 = [((i+1)==j) * one(UncF) for i in 1:2:2n, j in 1:2n] # pick momenta indices
        LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n] # pick position indices
        Σ0 = ΣT = [(i==j) * Σobs[i]  for i in 1:n, j in 1:n]
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


        # for ℓ in eachindex(Q.tt)
        #     u = deepmat(gr.Mt⁺[ℓ])
        #    print(u)
        #    println(isposdef(u+u'))
        #    println(eigmin(u+u'))
        # end
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


##########################
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

    BB = Bridge.B(0, Paux)          # does not depend on time
    β = vec(Bridge.β(0,Paux))       # does not depend on time
    σ̃T = σ̃(0, Paux) # vanilla, no (possibly enclose with Matrix)
    dt = t[2] - t[1]
    oldtemp = (0.5*dt) * Bridge.outer(Lt[end] * σ̃T)
    if lowrank
        # TBA lowrank on σ̃T, and write into σ̃T
    end

    for i in length(t)-1:-1:1
        dt = t[i+1]-t[i]
        if implicit
            Lt[i] .= Lt[i+1]/(I - dt* BB)
        else
            Lt[i] .=  Lt[i+1] * (I + BB * dt)
        end
        temp = (0.5*dt) * Bridge.outer(Lt[i] * σ̃T)
        Mt⁺[i] .= Mt⁺[i+1] + oldtemp + temp
        oldtemp = temp
        μt[i] .= μt[i+1] + 0.5 * (Lt[i] + Lt[i+1]) * β * dt  # trapezoid rule
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
function slogρ!(x0deepv, Q, W,X,logpriormom,llout) # stochastic approx to log(ρ)
    x0 = deepvec2state(x0deepv)
    lltemp = simguidedlm_llikelihood!(LeftRule(), X, x0, W, Q; skip=sk)   #overwrites X
    llout .= ForwardDiff.value.(lltemp)

    sum(lltemp) + logpriormom(x0)
end
slogρ!(Q, W, X,logpriormom, llout) = (x) -> slogρ!(x, Q, W,X,logpriormom,llout)


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
    updatekernel:  can be :mala_pos, :mala_mom, :rmmala_pos
"""
function update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,
                sampler, Q::GuidedProposal!, δ, updatekernel,logpriormom)
    n = Q.target.n
    x0 = deepvec2state(x)
    P = Q.target
    llout = copy(ll)
    lloutᵒ = copy(ll)

    if sampler ==:sgd  # ADJUST LATER
        sample!(W, Wiener{Vector{StateW}}())
        cfg = ForwardDiff.GradientConfig(slogρ(Q, W, X), x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
        ForwardDiff.gradient!(∇x, slogρ(Q, W, X),x,cfg) # X gets overwritten but does not chan
        x .+= δ * mask .* ∇x
        obj = simguidedlm_llikelihood!(LeftRule(), X, deepvec2state(x), W, Q; skip=sk)
    end
    if sampler==:mcmc
        accinit = 0.0
        ll_incl0 = 0,0
        ll_incl0ᵒ = 0.0 # define because of scoping rules

        if updatekernel in [:mala_pos, :mala_mom]
            cfg = ForwardDiff.GradientConfig(slogρ!(Q, W, X,logpriormom,llout), x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
            ForwardDiff.gradient!(∇x, slogρ!(Q, W, X,logpriormom,llout),x,cfg) # X gets overwritten but does not change
            ll_incl0 = sum(llout)
            if updatekernel==:mala_pos
                mask = deepvec(State(1 .- 0*x0.q,  0*x0.p))
                stepsize = δ[1]
            elseif updatekernel==:mala_mom
                mask = deepvec(State(0*x0.q, 1 .- 0*x0.p))
                stepsize = δ[2]
            end
            mask_id = (mask .> 0.1) # get indices that correspond to positions or momenta
            xᵒ .= x .+ .5 * stepsize * mask.* ∇x .+ sqrt(stepsize) .* mask .* randn(length(x))                              # should be ".=" or just "="?
            cfgᵒ = ForwardDiff.GradientConfig(slogρ!(Q, W, Xᵒ,logpriormom,lloutᵒ), xᵒ, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
            ForwardDiff.gradient!(∇xᵒ, slogρ!(Q, W, Xᵒ,logpriormom,lloutᵒ),xᵒ,cfgᵒ) # Xᵒ gets overwritten but does not change
            ll_incl0ᵒ = sum(lloutᵒ)
            ndistr = MvNormal(d * n,sqrt(stepsize))
            accinit = ll_incl0ᵒ - ll_incl0 -
                      logpdf(ndistr,(xᵒ - x - .5*stepsize .* mask.* ∇x)[mask_id]) +
                     logpdf(ndistr,(x - xᵒ - .5*stepsize .* mask.* ∇xᵒ)[mask_id])
             # plotting
             #Pdeterm = MarslandShardlow(0.1, 0.1, 0.0, 0.0, P.n)
             #plotlandmarkpositions[](initSamplePath(0:0.01:0.1,x0),Pdeterm,x0.q,deepvec2state(xᵒ).q;db=.5)
        elseif updatekernel == :rmmala_pos
               cfg = ForwardDiff.GradientConfig(slogρ!(Q, W, X,logpriormom,llout), x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
               ForwardDiff.gradient!(∇x, slogρ!(Q, W, X,logpriormom,llout),x,cfg) # X gets overwritten but does not change
               ll_incl0 = sum(llout)
               mask = deepvec(State(1 .- 0*x0.q,  0*x0.p))
               stepsize = δ[1]
               mask_id = (mask .> 0.1) # get indices that correspond to positions or momenta
               dK = gramkernel(x0.q,P)  #reshape([kernel(x0.q[i]- x0.q[j],P) * one(UncF) for i in 1:n for j in 1:n], n, n)
               ndistr = MvNormal(stepsize*dK)
               xᵒ = copy(x)
               xᵒ[mask_id] = x[mask_id] .+ .5 * stepsize * dK * ∇x[mask_id] .+ rand(ndistr)
               cfgᵒ = ForwardDiff.GradientConfig(slogρ!(Q, W, Xᵒ,logpriormom,lloutᵒ), xᵒ, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
               ForwardDiff.gradient!(∇xᵒ, slogρ!(Q, W, Xᵒ,logpriormom,lloutᵒ),xᵒ,cfgᵒ) # Xᵒ gets overwritten but does not change
               ll_incl0ᵒ = sum(lloutᵒ)

               x0ᵒ = deepvec2state(xᵒ)
               dKᵒ = gramkernel(x0ᵒ.q,P) #reshape([kernel(x0ᵒ.q[i]- x0ᵒ.q[j],P) * one(UncF) for i in 1:n for j in 1:n], n, n)
               ndistrᵒ = MvNormal(stepsize*dKᵒ)  #     ndistrᵒ = MvNormal(stepsize*deepmat(Kᵒ))

               accinit = ll_incl0ᵒ - ll_incl0 -
                         logpdf(ndistr,xᵒ[mask_id] - x[mask_id] - .5*stepsize * dK * ∇x[mask_id]) +
                        logpdf(ndistrᵒ,x[mask_id] - xᵒ[mask_id] - .5*stepsize * dKᵒ * ∇xᵒ[mask_id])
                # plotting
                #Pdeterm = MarslandShardlow(0.1, 0.1, 0.0, 0.0, P.n)
                #plotlandmarkpositions[](initSamplePath(0:0.01:0.1,x0),Pdeterm,x0.q,deepvec2state(xᵒ).q;db=.5)
        elseif updatekernel == :rmmala_mom
             cfg = ForwardDiff.GradientConfig(slogρ!(Q, W, X,logpriormom,llout), x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
             ForwardDiff.gradient!(∇x, slogρ!(Q, W, X,logpriormom,llout),x,cfg) # X gets overwritten but does not change
             ll_incl0 = sum(llout)
             mask = deepvec(State(0*x0.q, 1 .- 0*x0.p))
             stepsize = δ[2]
             mask_id = (mask .> 0.1) # get indices that correspond to positions or momenta

             # proposal step
             dK = gramkernel(x0.q,P)
             inv_dK = inv(dK)
             ndistr = MvNormal(stepsize*inv_dK)
             xᵒ = copy(x)
             xᵒ[mask_id] = x[mask_id] .+ .5 * stepsize * inv_dK * ∇x[mask_id] .+  rand(ndistr)

             # reverse step
             cfgᵒ = ForwardDiff.GradientConfig(slogρ!(Q, W, Xᵒ,logpriormom,lloutᵒ), xᵒ, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
             ForwardDiff.gradient!(∇xᵒ, slogρ!(Q, W, Xᵒ,logpriormom,lloutᵒ),xᵒ,cfgᵒ) # Xᵒ gets overwritten but does not change
             ll_incl0ᵒ = sum(lloutᵒ)
             ndistrᵒ = ndistr

             accinit = ll_incl0ᵒ - ll_incl0 -
                        logpdf(ndistr,xᵒ[mask_id] - x[mask_id] - .5*stepsize * inv_dK * ∇x[mask_id]) +
                       logpdf(ndistrᵒ,x[mask_id] - xᵒ[mask_id] - .5*stepsize * inv_dK * ∇xᵒ[mask_id])
         end

        # MH step
        if log(rand()) <= accinit
            #println("update initial state ", updatekernel, " accinit: ", round(accinit;digits=3), "  accepted")
            obj = ll_incl0ᵒ
            deepcopyto!(X, Xᵒ)
            x .= xᵒ
            ∇x .= ∇xᵒ
            ll .= lloutᵒ
            accepted = 1
        else
            #println("update initial state ", updatekernel, " accinit: ", round(accinit;digits=3), "  rejected")
            obj = ll_incl0
            #ll .= llout
            accepted = 0
        end
    end
    obj, (kernel = updatekernel, acc = accepted)
end

#---------------------------------------------------
# don't need these functions anymore
# logϕ(p) = -0.5 * norm(p)^2
# logϕ(qfix, p, P) = -hamiltonian(NState(qfix,p),P)
# function hamiltonian(x::NState, P::MarslandShardlow)
#     s = 0.0
#     for i in 1:P.n, j in 1:P.n
#         s += dot(x.p[i], x.p[j])*kernel(x.q[i] - x.q[j], P)
#     end
#     0.5 * s
# end
#---------------------------------------------------

"""
    For fixed Wiener increments and initial state, update parameters by random-walk-MH
"""
function update_pars!(obs_info,X, Xᵒ,W, Q, Qᵒ, x, ll, (prior_a, prior_c, prior_γ), (σ_a,σ_c,σ_γ))
    P = Q.target
    tt_ = Q.tt

    aᵒ = P.a * exp(σ_a * randn())
    cᵒ = P.c * exp(σ_c * randn())
    γᵒ = getγ(P) * exp(σ_γ * randn())
    putpars!(Qᵒ,(aᵒ,cᵒ,γᵒ))
    Pᵒ = Qᵒ.target
    update_guidrec!(Qᵒ, obs_info)   # compute backwards recursion


    llᵒ = simguidedlm_llikelihood!(LeftRule(), Xᵒ, deepvec2state(x), W, Qᵒ; skip=sk)
    A = logpdf(prior_a,aᵒ) - logpdf(prior_a,P.a) +
        logpdf(prior_c,cᵒ) - logpdf(prior_c,P.c) +
        logpdf(prior_γ,γᵒ) - logpdf(prior_γ,getγ(P)) +
        sum(llᵒ) - sum(ll) +
        logpdf(LogNormal(log(Pᵒ.a),σ_a),P.a)- logpdf(LogNormal(log(P.a),σ_a),Pᵒ.a)+
        logpdf(LogNormal(log(Pᵒ.c),σ_c),P.c)- logpdf(LogNormal(log(P.c),σ_c),Pᵒ.c)+
        logpdf(LogNormal(log(getγ(Pᵒ)),σ_γ),getγ(P))- logpdf(LogNormal(log(getγ(P)),σ_γ),getγ(Pᵒ))

    if log(rand()) <= A
        #println("logaccept for parameter update ", round(A;digits=4), "  accepted")
        ll .= llᵒ
        deepcopyto!(Q.guidrec,Qᵒ.guidrec)
        Q.target = Qᵒ.target
        deepcopyto!(Q.aux,Qᵒ.aux)
        deepcopyto!(X,Xᵒ)
        accept = 1
    else
        #println("logaccept for parameter update ", round(A;digits=4), "  rejected")
        accept = 0
    end
    (kernel = "parameterupdate", acc = accept)
end

"""
    Perform mcmc or sgd for landmarks model using the LM-parametrisation
    tt_:      time grid
    (xobs0,xobsT): observations at times 0 and T (at time T this is a vector)
    Σobs: covariance matrix of Gaussian noise assumed on each element of xobs0 and xobsT
    mT: vector of momenta at time T used for constructing guiding term
    P: target process

    sampler: either sgd (stochastic gradient descent) or mcmc (Markov Chain Monte Carlo)
    obs_atzero: Boolean, if true there is an observation at time zero
    fixinitmomentato0: Boolean, if true we assume at time zero we observe zero momenta
    xinit: initial guess on starting state

    ITER: number of iterations
    subsamples: vector of indices of iterations that are to be saved

    ρ: Crank-Nicolson parameter (ρ=0 is independence sampler)
    δ: step size for MALA updates on initial state (first component for positions, second component for momenta)
    prior_a: prior on parameter a
    prior_c: prior on parameter c
    prior_γ: prior on parameter γ
    σ_a: parameter determining update proposal for a [update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())]
    σ_c: parameter determining update proposal for c [update c to cᵒ as cᵒ = c * exp(σ_c * rnorm())]
    σ_γ: parameter determining update proposal for γ [update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())]
    η: (function) scheme to adapt tuning pars with diminishing adaptation

    initstate_updatetypes: specify type of updates on initialstate
    adaptskip: adapt mcmc tuning pars every adaptskip iteration

    outdir: output directory for animation
    pb:: Lmplotbounds (axis used for plotting landmarks evolution)
    updatepars: logical flag for updating pars a, c, γ
    make_animation: logical flag for making animation

    Returns:
    Xsave: saved iterations of all states at all times in tt_
    parsave: saved iterations of all parameter updates ,
    objvals: saved values of stochastic approximation to loglikelihood
    perc_acc: acceptance percentages (bridgepath - inital state)
"""
function lm_mcmc(tt_, (xobs0,xobsT), Σobs, mT, P,
         sampler, obs_atzero, fixinitmomentato0,
         xinit, ITER, subsamples,
        (ρinit, maxnrpaths, δinit, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ,η),
        logpriormom, initstate_updatetypes, adaptskip,
        outdir, pb; updatepars = true, make_animation=false)

    lt = length(tt_)
    StateW = PointF
    dwiener = dimwiener(P)
    # initialisation
    xobs0, obs_info = set_obsinfo(P.n,obs_atzero,fixinitmomentato0,Σobs,xobs0)
    nshapes = length(xobsT)
    guidrec = [init_guidrec(tt_,obs_info,xobs0) for k in 1:nshapes]  # memory allocation for backward recursion for each shape
    Paux = [auxiliary(P, State(xobsT[k],mT)) for k in 1:nshapes] # auxiliary process for each shape
    Q = GuidedProposal!(P,Paux,tt_,guidrec,xobs0,xobsT,nshapes,mT)
    update_guidrec!(Q, obs_info)   # compute backwards recursion

    X = [initSamplePath(tt_, xinit) for k in 1:nshapes]
    W = [initSamplePath(tt_,  zeros(StateW, dwiener)) for k in 1:nshapes]
    for k in 1:nshapes
        sample!(W[k], Wiener{Vector{StateW}}())
    end
    ll = simguidedlm_llikelihood!(LeftRule(), X, xinit, W, Q; skip=sk)

    # saving objects
    objvals = Float64[]             # keep track of (sgd approximation of the) loglikelihood
    Xsave = typeof(zeros(length(tt_) * P.n * 2 * d * nshapes))[]
    parsave = Vector{Float64}[]
    push!(Xsave, convert_samplepath(X))
    push!(objvals, sum(ll))
    push!(parsave, getpars(Q))

    # memory allocations
    Xᵒ = deepcopy(X)
    Qᵒ = deepcopy(Q)
    Wᵒ = initSamplePath(tt_,  zeros(StateW, dwiener))
    Wnew = initSamplePath(tt_,  zeros(StateW, dwiener))
    x = deepvec(xinit)
    xᵒ = deepcopy(x)
    ∇x = deepcopy(x)
    ∇xᵒ = deepcopy(x)

    if d==2
        xobs0comp1 = extractcomp(xobs0,1)
        xobs0comp2 = extractcomp(xobs0,2)
        xobsTcomp1 = extractcomp(xobsT[1],1)
        xobsTcomp2 = extractcomp(xobsT[1],2)
        pp1 = plotshapes(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2)
    else
        pp1 = Plots.plot()
    end

    accinfo = []                        # keeps track of accepted parameter and initial state updates
    accpcn = Int64[]                      # keeps track of nr of accepted pCN updates
    obj = 0

    δ = deepcopy(δinit)
    ρ = deepcopy(ρinit)

    anim = @animate for i in 1:ITER
        if make_animation
            drawpath(ITER, i-1,P.n,x,X[1],objvals,parsave,(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2),pb)
        end
        println();  println("iteration $i")

        # updates paths
        accpcn = update_path!(X, Xᵒ, W, Wᵒ, Wnew, ll, x, Q, ρ, accpcn, maxnrpaths)
        ρ = adaptpcnstep(i, accpcn, ρ, Q.nshapes, η; adaptskip = adaptskip)

        # update initial state
        for updatekernel in initstate_updatetypes
            obj, accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,sampler, Q, δ, updatekernel,logpriormom)
            push!(accinfo, accinfo_)
            if updatekernel in [:mala_mom, :rmmala_mom]
                δ[2] = adaptmalastep(i,accinfo,δ[2], η, updatekernel; adaptskip = adaptskip)
            else updatekernel in [:mala_pos, :rmmala_pos]
                δ[1] = adaptmalastep(i,accinfo,δ[1], η, updatekernel; adaptskip = adaptskip)
            end
        end


        # update parameters
        if updatepars
            accinfo_ = update_pars!(obs_info,X, Xᵒ,W, Q, Qᵒ, x, ll, (prior_a, prior_c, prior_γ), (σ_a,σ_c,σ_γ))
            push!(accinfo, accinfo_)

        (σ_a,σ_c,σ_γ) = adaptparstep(i,accinfo,(σ_a,σ_c,σ_γ), η;  adaptskip = adaptskip)
        end

        # update matching
        # direction, accinfo_ = update_matching(obs_info, X, Xᵒ,W, Q, Qᵒ, x, ll)
        # push!(accinfo, accinfo_)

        # adjust mT
        if mod(i,adaptskip)==0 && i < 100
            mTvec = [X[k][lt][2].p  for k in 1:nshapes]     # extract momenta at time T for each shape
            update_Paux_xT!(Q, mTvec, obs_info)
        end


        # save some of the results
        if i in subsamples
            push!(Xsave, convert_samplepath(X))
            push!(parsave, getpars(Q))
            push!(objvals, obj)
        end
        #

        println("ρ ", ρ ,",   δ ", δ, ",   σ_a ", σ_a)

        if make_animation && (i==ITER)
            drawpath(ITER, i, P.n,x,X[1],objvals,parsave,(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2),pb)
        end
    end
    anim, Xsave, parsave, objvals, accpcn, accinfo, δ, ρ
end

"""
    Adapt the step size for mala_mom updates.
    The adaptation is multiplication/divsion by exp(η(n))
"""
function adaptmalastep(n,accinfo,δ, η, updatekernel; adaptskip = 15, targetaccept=0.5)
    if mod(n,adaptskip)==0
        ind1 =  findall(first.(accinfo).== updatekernel)[end-adaptskip+1:end]
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
function adaptparstep(n,accinfo,(σ_a,σ_c,σ_γ), η; adaptskip = 15, targetaccept=0.5)
    if mod(n,adaptskip)==0
        ind1 =  findall(first.(accinfo).=="parameterupdate")[end-adaptskip+1:end]
        recent_mean = mean(last.(accinfo)[ind1])
        if recent_mean > targetaccept
            σ_a *= exp(η(n))
            σ_c *= exp(η(n))
            σ_γ *= exp(η(n))
        else
            σ_a *= exp(-η(n))
            σ_c *= exp(-η(n))
            σ_γ *= exp(-η(n))
        end
    end
    (σ_a,σ_c,σ_γ)
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
            #ρ = sigmoid(invsigmoid(ρ) * exp(-η(n)))
            ρ = sigmoid(invsigmoid(ρ) - η(n))
        else
            #ρ = sigmoid(invsigmoid(ρ) * exp(η(n)))
            ρ = sigmoid(invsigmoid(ρ) + η(n))
        end
    end
    ρ
end

### update_matching

function update_matching(obs_info,X, Xᵒ,W, Q, Qᵒ, x, ll)
    direction = rand(Bool)
    Qᵒ.xobsT = circshift.(Q.xobsT,direction)
    update_guidrec!(Qᵒ, obs_info)   # compute backwards recursion
    llᵒ = simguidedlm_llikelihood!(LeftRule(), Xᵒ, deepvec2state(x), W, Qᵒ; skip=sk)

    A = sum(llᵒ) - sum(ll)

    if log(rand()) <= A
        ll .= llᵒ
        deepcopyto!(Q.guidrec,Qᵒ.guidrec)
        #Q.xobsT = Qᵒ.xobsT
        deepcopyto!(Q.xobsT,Qᵒ.xobsT)
        deepcopyto!(X,Xᵒ)
        accept = 1
    else
        accept = 0
    end
    direction, (kernel = "matchingupdate", acc = accept)
end



#import Base: circshift
#circshift(x::NState) = NState(circshift(x.x,(0,1)))

"""
    Circular shift.
    Move right if direction==true, else move left.
"""
function circshift(x::Vector,direction::Bool)
    intdir = 2 * Int(direction) - 1
     x[circshift(1:length(x),intdir)]
end
