function simguidedlm_llikelihoodθ!(::LeftRule,  Xᵒ, x0, W, Q::GuidedProposal!, k; skip = 0, ll0 = true)
    Pnt = eltype(x0)
    tt =  Xᵒ.tt
    Xᵒ.yy[1] .= deepvalue(x0)

    typeθel = typeof(Q.target.a)
    som::typeθel  = 0.  # here is the only change!!!

    dump(typeθel)
    # initialise objects to write into
    # srout and strout are vectors of Points
    dwiener = dimwiener(Q.target)
    srout = zeros(Pnt, dwiener)
    strout = zeros(Pnt, dwiener)

    #x = copy(x0)  ##### ADJUST: this should also be of dual type !!!
    n = Q.target.n
    x = NState(zeros(Point{typeθel},n),zeros(Point{typeθel},n))

    xarb = NState(zeros(Point{typeθel},n),zeros(Point{typeθel},n))
    rout = copy(xarb)
    bout = copy(xarb)
    btout = copy(xarb)
    wout = copy(xarb)

    if !constdiff(Q)
        At = Bridge.a((1,0), x0, auxiliary(Q,k))  # auxtimehomogeneous switch
        A = zeros(Unc{deepeltype(x0)}, 2Q.target.n,2Q.target.n)
    end

    for i in 1:length(tt)-1
        dt = tt[i+1]-tt[i]
        #println(typeof(bout)==typeof(x))
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
                som += 0.5*(dot(At,Q.guidrec.Ht[i]) - dot(A,Q.guidrec.Ht[i])) * dt
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

function simguidedlm_llikelihoodθ!(::LeftRule,  X, x0, W, Q::GuidedProposal!; skip = 0, ll0 = true) # rather would like to dispatch on type and remove '_mv' from function name
    soms  = zeros(typeof(Q.target.a), Q.nshapes)
    for k in 1:Q.nshapes
        soms[k] = simguidedlm_llikelihoodθ!(LeftRule(), X[k],x0,W[k],Q,k ;skip=skip,ll0=ll0)
    end
    soms
end


# attempt to include automatic differentiation on parameter updates

function slogρ!(x0deepv, θ, Q, W, X, obs_info, llout)
    #(a, c, γ) = θ
    a = θ[1]; c= θ[2]; γ = θ[3]
    println("θ in slogρ! ")
    dump(a)
    P = Q.target
    if isa(P,MarslandShardlow)
        #Ptemp = MarslandShardlow(a, c, γ, P.λ, P.n)
        Ptemp = MarslandShardlow(a, c, γ, 0.0a, P.n) # be careful, only works if λ=0 is assumed
    elseif isa(P,Landmarks)
        nfs = construct_nfs(P.db, P.nfstd, γ)
        Ptemp = Landmarks(a, c, P.n, P.db, P.nfstd, nfs)
    end
    #Paux = [auxiliary(Ptemp,Q.xobsT[k]) for k in 1:Q.nshapes]

    Ptempaux = [auxiliary(Ptemp,State(Q.xobsT[k],Q.mT)) for k in 1:Q.nshapes] # auxiliary process for each shape
    Qtemp = GuidedProposal!(Ptemp,Ptempaux,Q.tt,Q.guidrec,Q.xobs0,Q.xobsT,Q.nshapes,Q.mT)
    update_guidrec!(Qtemp, obs_info)   # compute backwards recursion

    x0 = deepvec2state(x0deepv)
    lltemp = simguidedlm_llikelihoodθ!(LeftRule(), X, x0, W, Qtemp; skip=sk)  #overwrites X


    llout .= ForwardDiff.value.(lltemp)
    sum(lltemp)
end
slogρ!(x0deepv, Q, W, X, obs_info, llout) = (θ) -> slogρ!(x0deepv,θ, Q, W,X,obs_info,llout)

llout = copy(ll)
θ = [0.2, 0.4, 1.2]
slogρ!(deepvec(x0), θ, Q, W, X, obs_info, llout)

∇θ = zeros(3)
ForwardDiff.gradient!(∇θ, slogρ!(deepvec(x0), Q, W, X,obs_info,llout),θ) # X gets overwritten but does not change

ForwardDiff.gradient(slogρ!(deepvec(x0), Q, W, X,obs_info,llout),θ) # X gets overwritten but does not change
