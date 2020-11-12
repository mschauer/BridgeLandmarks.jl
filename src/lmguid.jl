import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

"""
    lm_mcmc(t, obsinfo, mT, P, ITER, subsamples, xinit, pars, priorθ, priormom, updatescheme, outdir)

This is the main function call for doing either MCMC or SGD for landmark models
Backward ODEs used are in terms of the LMμ-parametrisation

## Arguments
- `t`:      time grid
- `obsinfo`: struct of type `ObsInfo`
- `mT`: vector of momenta at time T used for constructing guiding term
- `P`: target process
- `ITER`: number of iterations
- `subsamples`: vector of indices of iterations that are to be saved
- `xinit`: initial value on state at time 0 (`x0`)
- `pars`: tuning pars for updates (see `?Pars_ms()` and `?Pars_ahs()` for documentation)
- `priorθ`: prior on θ = (a,c,γ)
- `priormom`: prior on initial momenta
- `updatescheme`: vector specifying mcmc-updates
- `outdir` output directory for animation

## Returns:
- `Xsave`: saved iterations of all states at all times in tt_
- `parsave`: saved iterations of all parameter updates ,
- `accinfo`: acceptance indicators of mcmc-updates
- `δ`: value of `(δpos, δmom)` af the final iteration of the algorithm (these are stepsize parameters for updating initial positions and momenta respectively)
- `ρ`: value of `ρinit`  af the final iteration of the algorithm
- `covθprop`: value of `covθprop` at the final iteration of the algorithm
"""
function lm_mcmc(t, obsinfo, mT, P, ITER, subsamples, xinit, pars, priorθ, priormom, updatescheme, outdir, printskip)
    lt = length(t)
    StateW = PointF
    dwiener = dimwiener(P)
    nshapes = obsinfo.nshapes

    guidrec = [GuidRecursions(t,obsinfo)  for _ ∈ 1:nshapes]  # initialise guiding terms
    Paux = [auxiliary(P, State(obsinfo.xobsT[k],mT)) for k ∈ 1:nshapes] # auxiliary process for each shape
    Q = GuidedProposal(P,Paux,t,obsinfo.xobs0,obsinfo.xobsT,guidrec,nshapes,[mT for _ ∈ 1:nshapes])
    Q = update_guidrec!(Q, obsinfo)   # compute backwards recursion

    X = [initSamplePath(t, xinit) for _ ∈ 1:nshapes]
    W = [initSamplePath(t,  zeros(StateW, dwiener)) for _ ∈ 1:nshapes]
    for k ∈ 1:nshapes   sample!(W[k], Wiener{Vector{StateW}}())  end

    x = deepvec(xinit)
    q, p = split_state(xinit)
    qᵒ = deepcopy(q); pᵒ = deepcopy(q); ∇ = deepcopy(q); ∇ᵒ = deepcopy(q)

    # memory allocations, actual state at each iteration is (X,W,Q,x,∇x) (x, ∇x are initial state and its gradient)
    Xᵒ = deepcopy(X)
    Wᵒ = initSamplePath(t,  zeros(StateW, dwiener))
    Wnew = initSamplePath(t,  zeros(StateW, dwiener))
    # sample guided proposal and compute loglikelihood (write into X)
    ll, X = gp!(LeftRule(), X, xinit, W, Q; skip=sk)

    # setup containers for saving objects
    Xsave = typeof(zeros(length(t) * P.n * 2 * d * nshapes))[]
    push!(Xsave, convert_samplepath(X))
    parsave = Vector{Float64}[]
    push!(parsave, getpars(Q))
    initendstates_save = [extract_initial_and_endstate(0,X[1]) ]
    accinfo = DataFrame(fill([], length(updatescheme)+1), [updatescheme...,:iteration]) # columns are all update types + 1 column for iteration number
    acc = zeros(ncol(accinfo))

    δ = [pars.δpos, pars.δmom, pars.δsgd_mom]
    δa, δγ = pars.δa, pars.δγ
    ρ = pars.ρlowerbound
    adapt(i) = (i > 1.5*pars.adaptskip) & (mod(i,pars.adaptskip)==0)

    x0 = X[1].yy[1]
    dK = gramkernel(x0.q, P)
    inv_dK = inv(dK)

    for i ∈ 1:ITER
        k = 1
        for update ∈ updatescheme
            if update == :innov
                accinfo_, X, W, ll = update_path!(X, W, ll, Xᵒ, Wᵒ, Wnew, Q, ρ)
                if adapt(i)
                    ρ = adaptpcnstep(ρ, i, accinfo[!,update], pars.η, pars.adaptskip)
                end
            elseif update ∈ [:mala_mom, :rmmala_mom, :rmrw_mom]
                accinfo_, X, x, ∇, ll = update_initialstate!(X,Xᵒ,W,ll, x, qᵒ, pᵒ,∇, ∇ᵒ, Q, δ, update, priormom, (dK, inv_dK))
                if adapt(i)
                    δ[2] = adapt_pospar_step(δ[2], i, accinfo[!,update], pars.η, pars.adaptskip)
                end
            elseif update ∈ [:mala_pos, :rmmala_pos]
                accinfo_ , X, x, ∇, ll= update_initialstate!(X,Xᵒ,W,ll, x, qᵒ, pᵒ,∇, ∇ᵒ, Q, δ, update, priormom, (dK, inv_dK))
                if adapt(i)
                    δ[1] = adapt_pospar_step(δ[1], i, accinfo[!,update], pars.η, pars.adaptskip)
                end
            elseif update == :parameter
                Q, accinfo_, X, ll = update_pars!(X, ll, Xᵒ,W, Q, priorθ, obsinfo, δa, δγ)
                if adapt(i)
                    δa = adapt_pospar_step(δa, i, accinfo[!,update], pars.η, pars.adaptskip)
                end
            elseif update == :matching
                obsinfo, accinfo_, Q, X, ll = update_cyclicmatching(X, ll, obsinfo, Xᵒ, W, Q)
            elseif update == :sgd_mom
                accinfo_, X, x, ∇, ll = update_initialstate!(X,Xᵒ,W,ll, x, qᵒ, pᵒ,∇, ∇ᵒ, Q, δ, update, priormom, (dK, inv_dK))
            end
            acc[k] = accinfo_
            k += 1
        end
        acc[end] = i
        push!(accinfo, acc)

        if mod(i,pars.adaptskip)==0 && i < 100  # adjust mT (the momenta at time T used in the construction of the guided proposal)
            mTvec = [X[k][lt][2].p  for k in 1:nshapes]     # extract momenta at time T for each shape
    #       Q = update_mT!(Q, mTvec, obsinfo)
        end

        # save some of the results
        if i ∈ subsamples
            push!(Xsave, convert_samplepath(X))
        end
        push!(parsave, getpars(Q))
        push!(initendstates_save, extract_initial_and_endstate(i,X[1]))

        if mod(i,printskip) == 0
            #println();  println("iteration $i, ρ = $ρ,  δ = $δ,  δa = $δa")
            #print("Average acceptance rates over last ",  printskip, " iterations: ")
            if i >= pars.adaptskip
                ac = accinfo[end-pars.adaptskip+1:end, 1:end-1]
                println(round.([mean(x) for x in eachcol(ac)];digits=2))
            end
            #println("parameter a and γ equal: ", getpars(Q))
        end
    end
    Xsave, parsave, initendstates_save, accinfo, δ, ρ, δa
end

"""
    update_path!(X,Xᵒ,W,Wᵒ,Wnew,ll,x, Q, ρ)

Update bridges for all shapes using Crank-Nicholsen scheme with parameter `ρ` (only in case the method is mcmc).
Newly accepted bridges are written into `(X,W)`, loglikelihood on each segment is written into vector `ll`
All Wiener increments are always updated.

## Write into
- `X`: diffusion paths
- `W`: innovations
- `ll`: vector of loglikehoods (k-th element is for the k-th shape)
"""
function update_path!(X, W, ll, Xᵒ,Wᵒ, Wnew, Q, ρlowerbound)
    acc = Int64[]
    x0 = X[1].yy[1]
    ρ = rand(Uniform(ρlowerbound, 1.0))
    ρ_ = sqrt(1.0-ρ^2)
    # From current state (x,W) with loglikelihood ll, update to (x, Wᵒ)
    for k ∈ 1:Q.nshapes
        sample!(Wnew, Wiener{Vector{PointF}}())
        for i ∈ eachindex(W[1].yy)
            Wᵒ.yy[i] = ρ * W[k].yy[i] + ρ_ * Wnew.yy[i]
        end
        llᵒ, Xᵒ[k] = gp!(LeftRule(), Xᵒ[k], x0, Wᵒ, Q, k; skip=sk)
        diff_ll = llᵒ - ll[k]
        if log(rand()) <= diff_ll
            for i ∈ eachindex(W[1].yy)
                X[k].yy[i] .= Xᵒ[k].yy[i]
                W[k].yy[i] .= Wᵒ.yy[i]
            end
            ll[k] = llᵒ
            push!(acc,1)
        else
            push!(acc,0)
        end
    end
    mean(acc), X, W, ll
end



"""
    slogρ!(q, p, Q, W, X, priormom,llout)

Main use of this function is to get gradient information of the loglikelihood with respect ot the initial state.

## Arguments
- `q` and `p` make up the initial state (which could be obtained from `x0 = merge_state(q,p)`)
- `Q`: GuidePropsoal!
- `W`: vector of innovations
- `X`: sample path
- `priormom`: prior on the initial momenta
- `llout`: vector where loglikelihoods for each shape are written into

## Writes into
- The guided proposal is written into `X`.
- The computed loglikelihood for each shape is written into `llout`

## Returns
- loglikelihood (summed over all shapes) + the logpriordensity of the initial momenta
"""
function slogρ!(q,p, Q, W, X, priormom, llout)
    lltemp, X = gp!(LeftRule(), X, q, p, W, Q; skip=sk)   # writes into X
    llout .= ForwardDiff.value.(lltemp)
    sum(lltemp) + logpdf(priormom, p)
end

"""
    slogρ_pos!(p, Q, W, X,priormom, llout) = (q) -> slogρ!(q, p , Q, W, X, priormom, llout)
"""
slogρ_pos!(p, Q, W, X,priormom, llout) = (q) -> slogρ!(q, p , Q, W, X, priormom, llout)


"""
    slogρ_mom!(q, Q, W, X,priormom, llout) = (p) -> slogρ!(q, p , Q, W, X, priormom, llout)
"""
slogρ_mom!(q, Q, W, X,priormom, llout) = (p) -> slogρ!(q, p , Q, W, X, priormom, llout)


"""
    update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇, ∇ᵒ, Q::GuidedProposal, δ, update,priormom, (dK, inv_dK))

## Arguments
- `X`:  current iterate of vector of sample paths
- `Xᵒ`: vector of sample paths to write proposal into
- `W`:  current vector of Wiener increments
- `ll`: current value of loglikelihood
- `x`, `xᵒ`, `∇x`, `∇xᵒ`: allocated vectors for initial state and its gradient (both actual and proposed values)
- `sampler`: either sgd (not checked yet) or mcmc
- `Q`: GuidedProposal
- `δ`: vector with MALA stepsize for initial state positions (δ[1]) and initial state momenta (δ[2])
- `update`:  can be `:mala_pos`, `:mala_mom`, :rmmala_pos`, `:rmmala_mom`, ``:rmrw_mom`
- `priormom`: prior on initial momenta
- (dK, inv_dK): gramkernel at x0.q and its inverse

## Writes into / modifies
- `X`: landmarks paths
- `x`: initial state of the landmark paths
- ``∇`: gradient of loglikelihood with respect to initial state `x`
- `ll`: a vector with as k-th element the loglikelihood for the k-th shape

## Returns
-  0/1 indicator (reject/accept)
"""
function update_initialstate!(X,Xᵒ,W,ll, x, qᵒ, pᵒ,∇, ∇ᵒ,
                 Q::GuidedProposal, δ, update, priormom, (dK, inv_dK))

    P = Q.target;   n = P.n
    x0 = deepvec2state(x)
    q, p = split_state(x0)
    dn = d * n

    if update ==:sgd_mom
        # @warn "Option :sgd_mom has not been checked/tested so far; don't use as yet."
        for k ∈ 1:Q.nshapes
            sample!(W[k], Wiener{Vector{PointF}}())
        end
        u = slogρ_mom!(q, Q, W, X, priormom,ll)
        cfg = ForwardDiff.GradientConfig(u, p, ForwardDiff.Chunk{dn}()) # d*P.n is maximal
        ForwardDiff.gradient!(∇, u, p, cfg) # X gets overwritten but does not change
        pᵒ = p .- δ[3] * ∇ # trun(∇, δ[3])
        x0ᵒ = merge_state(q, pᵒ)
        ll, X = gp!(LeftRule(), X, x0ᵒ, W, Q)
        x .= deepvec(x0ᵒ)
        accepted = 1.0
    else
        accinit = 0.0
        llᵒ = copy(ll)
        if update in [:mala_pos, :rmmala_pos]
            u = slogρ_pos!(p, Q, W, X, priormom,ll)
            uᵒ = slogρ_pos!(p, Q, W, Xᵒ, priormom,llᵒ)
            cfg = ForwardDiff.GradientConfig(u, q, ForwardDiff.Chunk{dn}()) # d*P.n is maximal
            ForwardDiff.gradient!(∇, u, q, cfg) # X and ll get overwritten but do not change
            stepsize = sample(δ[1])
        end
        if update in [:mala_mom, :rmmala_mom]
            u = slogρ_mom!(q, Q, W, X, priormom,ll)
            uᵒ = slogρ_mom!(q, Q, W, Xᵒ, priormom,llᵒ)
            cfg = ForwardDiff.GradientConfig(u, p, ForwardDiff.Chunk{dn}()) # d*P.n is maximal
            ForwardDiff.gradient!(∇, u, p, cfg) # X and ll get overwritten but do not change
            #stepsize = δ[2]
            stepsize = sample(δ[2])
        end
        if update == :rmrw_mom
            u = slogρ_mom!(q, Q, W, X, priormom,ll)
            uᵒ = slogρ_mom!(q, Q, W, Xᵒ, priormom,llᵒ)
            #stepsize = δ[2]
            stepsize = sample(δ[2])
        end
        if update == :mala_pos
            ndistr = MvNormal(dn,sqrt(stepsize))
            qᵒ .= q .+ .5 * stepsize *  ∇ .+ rand(ndistr)
            cfgᵒ = ForwardDiff.GradientConfig(uᵒ, qᵒ, ForwardDiff.Chunk{dn}())
            ForwardDiff.gradient!(∇ᵒ, uᵒ, qᵒ, cfgᵒ)
            accinit = sum(llᵒ) - sum(ll) -
                      logpdf(ndistr, qᵒ - q - .5*stepsize * ∇) +
                      logpdf(ndistr, q - qᵒ - .5*stepsize * ∇ᵒ)
        elseif update == :mala_mom
            ndistr = MvNormal(dn, sqrt(stepsize))
            pᵒ .= p .+ .5 * stepsize * trun(∇,stepsize) .+ rand(ndistr)
            cfgᵒ = ForwardDiff.GradientConfig(uᵒ, pᵒ, ForwardDiff.Chunk{dn}())
            ForwardDiff.gradient!(∇ᵒ, uᵒ, pᵒ, cfgᵒ)
            accinit = sum(llᵒ) - sum(ll) -
                      logpdf(ndistr, pᵒ - p - .5*stepsize * trun(∇,stepsize)) +
                      logpdf(ndistr, p - pᵒ - .5*stepsize * trun(∇ᵒ,stepsize))
                      #logpdf(priormom, pᵒ) - logpdf(priormom, p)
        elseif update == :rmmala_pos
            #dK = gramkernel(x0.q, P)
            dK = gramkernel(reinterpret(PointF,q), P)
            ndistr = MvNormal(zeros(d*n),stepsize*dK)
            qᵒ .= q .+ .5 * stepsize * dK * trun(∇,stepsize) .+ rand(ndistr)
            cfgᵒ = ForwardDiff.GradientConfig(uᵒ, qᵒ, ForwardDiff.Chunk{dn}())
            ForwardDiff.gradient!(∇ᵒ, uᵒ, qᵒ, cfgᵒ)
            dKᵒ = gramkernel(reinterpret(PointF,qᵒ), P)
            ndistrᵒ = MvNormal(zeros(d*n),stepsize*dKᵒ)
            accinit = sum(llᵒ) - sum(ll) -
                     logpdf(ndistr, qᵒ - q - .5*stepsize * dK * trun(∇,stepsize)) +
                     logpdf(ndistrᵒ, q - qᵒ - .5*stepsize * dKᵒ * trun(∇ᵒ,stepsize))
        elseif update == :rmmala_mom
             ndistr = MvNormal(stepsize*inv_dK)
             pᵒ .= p .+ .5 * stepsize * inv_dK * ∇ .+  rand(ndistr)
             cfgᵒ = ForwardDiff.GradientConfig(uᵒ, pᵒ, ForwardDiff.Chunk{dn}())
             ForwardDiff.gradient!(∇ᵒ, uᵒ, pᵒ, cfgᵒ) # Xᵒ gets overwritten but does not change
             accinit = sum(llᵒ) - sum(ll) -
                       logpdf(ndistr, pᵒ - p - .5*stepsize * inv_dK * ∇) +
                       logpdf(ndistr, p - pᵒ - .5*stepsize * inv_dK * ∇ᵒ)
       elseif update == :rmrw_mom
            ndistr = MvNormal(zeros(d*n),stepsize*inv_dK)
            pᵒ .= p  .+ rand(ndistr)
            uᵒ(pᵒ)  # writes into llᵒ, don't need gradient info here
            accinit = sum(llᵒ) - sum(ll)   # proposal is symmetric
        end

        # MH acceptance decision
        if log(rand()) <= accinit
            X = copypaths!(X,Xᵒ)
            ll .= llᵒ
            if update in [:mala_pos, :rmmala_pos]
                x .= deepvec(merge_state(qᵒ, p))
            end
            if update in [:mala_mom, :rmmala_mom, :rmrw_mom]
                x .= deepvec(merge_state(q, pᵒ))
            end
            accepted = 1.0
        else
            accepted = 0.0
        end
    end
    accepted, X, x, ∇, ll
end

"""
    trun(x,h) = x/max(1.0,0.5*h*norm(x))

useful function for truncated version of mala
"""
trun(x,h) = x/max(1.0,0.5*h*norm(x))


"""
    update_pars!(X, Xᵒ,W, Q, x, ll, priorθ, covθprop, obsinfo)

For fixed Wiener increments and initial state, update parameters by random-walk-MH.

## Writes into / modifies
`X`: landmark paths under landmarks process with parameter θ
`ll`: vector of loglikelihoods (one for each shape)

## Returns
`Q, accept`: where `accept` is 1/0 according to accept/reject in the MH-step.
"""
function update_pars!(X, ll, Xᵒ, W, Q , priorθ,  obsinfo, δa, δγ)
    θ = getpars(Q)
    a = θ[1];  γ = θ[2]
    aᵒ = a * exp(δa*randn())
    γᵒ = γ * exp(δγ*randn())
    θᵒ = [aᵒ, γᵒ]
    Qᵒ = adjust_to_newpars(Q, θᵒ, obsinfo)
    x0 = X[1].yy[1]
    llᵒ, Xᵒ = gp!(LeftRule(), Xᵒ,x0, W, Qᵒ; skip=sk)

    A = sum(llᵒ) - sum(ll) +
        logpdf(priorθ, θᵒ) - logpdf(priorθ, θ) +
        log(γᵒ) - log(γ) + log(aᵒ) - log(a)
    if log(rand()) <= A
        ll .= llᵒ
        X = copypaths!(X,Xᵒ)
        # @infiltrate
        for k ∈ 1:Q.nshapes
            for i ∈ eachindex(X[1].yy)
                X[k].yy[i] .= Xᵒ[k].yy[i]
            end
        end
        Q = Qᵒ
        #Q = deepcopy(Qᵒ)
        accept = 1.0
    else
        accept = 0.0
    end
    Q, accept, X, ll
end


"""
    function copypaths!(X,Xᵒ)

Write Xᵒ into X
"""
function copypaths!(X,Xᵒ)
    for k ∈ eachindex(X), i ∈ eachindex(X[1].yy)
        X[k].yy[i] .= Xᵒ[k].yy[i]
    end
    X
end



######### adaptation of mcmc tuning parameters #################

"""
    adapt_pospar_step!(δ, n, accinfo, η, update; adaptskip = 15, targetaccept=0.5)

Adapt the step size for updates, where tuning par has to be positive.
The adaptation is multiplication/divsion by exp(η(n)).
"""
function adapt_pospar_step(δ, n, accinfo, η, adaptskip; targetaccept=0.5)
    recent_mean = mean(accinfo[end-adaptskip+1:end])
    if recent_mean > targetaccept
        return δ * exp(η(n))
    else
        return δ * exp(-η(n))
    end
end


"""
    adaptpcnstep(n, accpcn, ρ, nshapes, η; adaptskip = 15, targetaccept=0.5)

Adjust pcN-parameter `ρ` adaptive every `adaptskip` steps.
For that `ρ` is first tranformed to (-∞, ∞), then updated, then mapped back to (0,1)
"""
function adaptpcnstep(ρ, n, accinfo, η, adaptskip; targetaccept=0.5)
    recentmean = mean(accinfo[end-adaptskip+1:end])
    if recentmean > targetaccept
        out =  logistic(logit(ρ) - η(n))
    else
        out =  logistic(logit(ρ) + η(n))
    end
    out
end

"""
    show_updates()

Convenience function to show the available implemted updates.
"""
function show_updates()
    println("Available implemented update steps are:")
    println(":innov")
    println(":sgd_mom")
    println(":mala_mom, :rmmala_mom, :rmrw_mom")
    println(":mala_pos, :rmmala_pos")
    println(":parameter, :matching")
end
