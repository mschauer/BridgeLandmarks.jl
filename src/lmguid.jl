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
function lm_mcmc(t, obsinfo, mT, P, ITER, subsamples, xinit, pars, priorθ, priormom, updatescheme, outdir)
    lt = length(t);   StateW = PointF;    dwiener = dimwiener(P);   nshapes = obsinfo.nshapes

    guidrec = [GuidRecursions(t,obsinfo)  for _ in 1:nshapes]  # initialise guiding terms
    Paux = [auxiliary(P, State(obsinfo.xobsT[k],mT)) for k in 1:nshapes] # auxiliary process for each shape
    Q = GuidedProposal(P,Paux,t,obsinfo.xobs0,obsinfo.xobsT,guidrec,nshapes,[mT for _ in 1:nshapes])
    update_guidrec!(Q, obsinfo)   # compute backwards recursion

    X = [initSamplePath(t, xinit) for _ in 1:nshapes]
    W = [initSamplePath(t,  zeros(StateW, dwiener)) for _ in 1:nshapes]
    for k in 1:nshapes   sample!(W[k], Wiener{Vector{StateW}}())  end

    x = deepvec(xinit)
    ∇x = deepcopy(x)

    # memory allocations, actual state at each iteration is (X,W,Q,x,∇x) (x, ∇x are initial state and its gradient)
    Xᵒ = deepcopy(X)
    Wᵒ = initSamplePath(t,  zeros(StateW, dwiener))
    Wnew = initSamplePath(t,  zeros(StateW, dwiener))
    xᵒ = deepcopy(x)
    ∇xᵒ = deepcopy(x)

    # sample guided proposal and compute loglikelihood (write into X)
    ll = gp!(LeftRule(), X, xinit, W, Q; skip=sk)

    # for Riemannian updates on the momenta, assuming initial point is fixed
    dK = gramkernel(xinit.q, P)
    invK = inv(dK)

    # setup containers for saving objects
    Xsave = typeof(zeros(length(t) * P.n * 2 * d * nshapes))[]
    push!(Xsave, convert_samplepath(X))
    parsave = Vector{Float64}[]
    push!(parsave, getpars(Q))
    accinfo = DataFrame(fill([], length(updatescheme)+1), [updatescheme...,:iteration]) # columns are all update types + 1 column for iteration number
    acc = zeros(ncol(accinfo))

    δ = [pars.δpos, pars.δmom]
    ρ = pars.ρinit
    covθprop = pars.covθprop
    askip = pars.adaptskip

    for i in 1:ITER
        k = 1
        for update in updatescheme
            if update == :innov
                #@timeit to "path update"
                accinfo_ = update_path!(X, W, ll, Xᵒ,Wᵒ, Wnew, Q, ρ)
                if (i > 2askip) & (mod(i,askip)==0)
                    ρ = adaptpcnstep(ρ, i, accinfo[!,update], pars.η;  adaptskip = askip)
                end
            elseif update in [:mala_mom, :rmmala_mom, :rmrw_mom]
                #@timeit to "update mom"
                accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,:mcmc, Q, δ, update, priormom, invK)
                if (i > 2askip) & (mod(i,askip)==0)
                    δ[2] = adaptmalastep(δ[2], i, accinfo[!,update], pars.η, update; adaptskip = askip)
                end
            elseif update in [:mala_pos, :rmmala_pos]
                #@timeit to "update pos"
                accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,:mcmc, Q, δ, update, priormom, invK)
                if (i > 2askip) & (mod(i,askip)==0)
                    δ[1] = adaptmalastep(δ[1], i, accinfo[!,update], pars.η, update; adaptskip = askip)
                end
            elseif update == :parameter
                #@timeit to "update par"
                Q, accinfo_ = update_pars!(X, ll, Xᵒ,W, Q, priorθ, covθprop, obsinfo)
                if (i > 2askip) & (mod(i,askip)==0)
                    covθprop = adaptparstep(covθprop, i, accinfo[!,update], pars.η; adaptskip = askip)
                end
                #@show Q.target
            elseif update==:matching
                Q, obsinfo, accinfo_ = update_cyclicmatching(X, ll, obsinfo, Xᵒ, W, Q)
            elseif update == :sgd
                accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,:sgd, Q, δ, update,priormom)
            end
            acc[k] = accinfo_
            k += 1
        end
        acc[end] = i
        push!(accinfo, acc)

        if mod(i,askip)==0 && i < 100  # adjust mT (the momenta at time T used in the construction of the guided proposal)
            mTvec = [X[k][lt][2].p  for k in 1:nshapes]     # extract momenta at time T for each shape
            update_mT!(Q, mTvec, obsinfo)
        end

        # save some of the results
        if i in subsamples
            push!(Xsave, convert_samplepath(X))
        end
        push!(parsave, getpars(Q))

        if mod(i,5) == 0
            println();  println("iteration $i")
            println("ρ ", ρ , ",   δ ", δ, ",   covθprop ", covθprop)
            println(round.([mean(x) for x in eachcol(accinfo[!,1:end-1])];digits=2))
        end
    end
    Xsave, parsave, accinfo, δ, ρ, covθprop
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
function update_path!(X, W, ll, Xᵒ,Wᵒ, Wnew, Q, ρ)
    acc = Int64[]
    x0 = X[1].yy[1]
    # From current state (x,W) with loglikelihood ll, update to (x, Wᵒ)
    for k in 1:Q.nshapes
        sample!(Wnew, Wiener{Vector{PointF}}())
        for i in eachindex(W[1].yy)
            Wᵒ.yy[i] = ρ * W[k].yy[i] + sqrt(1-ρ^2) * Wnew.yy[i]
        end
        llᵒ = gp!(LeftRule(), Xᵒ[k], x0, Wᵒ, Q, k; skip=sk)
        diff_ll = llᵒ - ll[k]
        if log(rand()) <= diff_ll
            for i in eachindex(W[1].yy)
                X[k].yy[i] .= Xᵒ[k].yy[i]
                W[k].yy[i] .= Wᵒ.yy[i]
            end
            ll[k] = llᵒ
            push!(acc,1)
        else
            push!(acc,0)
        end
    end
    mean(acc)
end

"""
    slogρ!(x0deepv, Q, W,X,priormom,llout)

Main use of this function is to get gradient information of the loglikelihood with respect ot the initial state.

## Arguments
- `x0deepv`: initial state, converted to a deepvector (i.e. all elements stacked)
- `Q`: GuidePropsoal!
- `X`: sample path
- `priormom`: prior on the initial momenta
- `llout`: vector where loglikelihoods for each shape are written into

## Writes into
- A guided proposal is simulated forward starting in `x0` which is the state corresponding to `x0deepv`. The guided proposal is written into `X`.
- The computed loglikelihood for each shape is written into `llout`

## Returns
- loglikelihood (summed over all shapes) + the logpriordensity of the initial momenta
"""
function slogρ!(x0deepv, Q, W, X, priormom,llout)
    x0 = deepvec2state(x0deepv)
    lltemp = gp!(LeftRule(), X, x0, W, Q; skip=sk)   #overwrites X
    llout .= ForwardDiff.value.(lltemp)
    sum(lltemp) + logpdf(priormom, vcat(p(x0)...))
end

"""
    slogρ!(Q, W, X,priormom, llout) = (x) -> slogρ!(x, Q, W,X,priormom,llout)
"""
slogρ!(Q, W, X,priormom, llout) = (x) -> slogρ!(x, Q, W,X,priormom,llout)

"""
    update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,sampler, Q::GuidedProposal, δ, update,priormom)

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

## Writes into / modifies
- `X`: landmarks paths
- `x`: initial state of the landmark paths
- ``∇x`: gradient of loglikelihood with respect to initial state `x`
- `ll`: a vector with as k-th element the loglikelihood for the k-th shape

## Returns
-  0/1 indicator (reject/accept)
"""
function update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,
                sampler, Q::GuidedProposal, δ, update, priormom, invK)

    @assert x==xᵒ  "x and xᵒ are not the same at start of update_initialstate!"
    P = Q.target
    n = P.n
    x0 = deepvec2state(x)
    llᵒ = copy(ll)
    u = slogρ!(Q, W, X, priormom,ll)
    uᵒ = slogρ!(Q, W, Xᵒ, priormom,llᵒ)
    cfg = ForwardDiff.GradientConfig(u, x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal
    ForwardDiff.gradient!(∇x, u, x, cfg) # X and ll get overwritten but do not change

    if sampler ==:sgd  # CHECK VALIDITY LATER
        @warn "Option :sgd has not been checked so far; don't use as yet."
        mask = deepvec(State(0*x0.q, onemask(x0.p)))
        # StateW = PointF
        # sample!(W, Wiener{Vector{StateW}}())
        ForwardDiff.gradient!(∇x, u, x, cfg) # X gets overwritten but does not chang
        xᵒ = x .+ δ[2] * mask .* ∇x
        slogρ!(Q, W, X, priormom, llout)(xᵒ)

        obj = sum(llout)
        #obj = gp!(LeftRule(), X, deepvec2state(x), W, Q; skip=sk)
        accepted = 1
    end
    if sampler==:mcmc
        accinit = 0.0
        if update in [:mala_pos, :rmmala_pos]
            mask = deepvec(State(onemask(x0.q),  0*x0.p))
            stepsize = δ[1]
        elseif update in [:mala_mom, :rmmala_mom, :rmrw_mom]
            mask = deepvec(State(0*x0.q, onemask(x0.p)))
            stepsize = δ[2]
        end
        mask_id = (mask .> 0.1) # get indices that correspond to positions or momenta
        if update in [:mala_pos, :mala_mom]
            xᵒ .= x .+ .5 * stepsize * mask.* ∇x .+ sqrt(stepsize) .* mask .* randn(length(x))
            cfgᵒ = ForwardDiff.GradientConfig(uᵒ, xᵒ, ForwardDiff.Chunk{2*d*n}())
            ForwardDiff.gradient!(∇xᵒ, uᵒ, xᵒ, cfgᵒ)
            ndistr = MvNormal(d * n,sqrt(stepsize))
            accinit = sum(llᵒ) - sum(ll) -
                      logpdf(ndistr,(xᵒ - x - .5*stepsize .* mask.* ∇x)[mask_id]) +
                     logpdf(ndistr,(x - xᵒ - .5*stepsize .* mask.* ∇xᵒ)[mask_id])
        elseif update == :rmmala_pos
            dK = gramkernel(x0.q,P)
            ndistr = MvNormal(zeros(d*n),stepsize*dK)
            xᵒ[mask_id] .= x[mask_id] .+ .5 * stepsize * dK * ∇x[mask_id] .+ rand(ndistr) # maybe not .=
            cfgᵒ = ForwardDiff.GradientConfig(uᵒ, xᵒ, ForwardDiff.Chunk{2*d*n}())
            ForwardDiff.gradient!(∇xᵒ, uᵒ, xᵒ, cfgᵒ)
            x0ᵒ = deepvec2state(xᵒ)
            dKᵒ = gramkernel(x0ᵒ.q,P)
            ndistrᵒ = MvNormal(zeros(d*n),stepsize*dKᵒ)  #     ndistrᵒ = MvNormal(stepsize*deepmat(Kᵒ))
            accinit = sum(llᵒ) - sum(ll) -
                     logpdf(ndistr,xᵒ[mask_id] - x[mask_id] - .5*stepsize * dK * ∇x[mask_id]) +
                    logpdf(ndistrᵒ,x[mask_id] - xᵒ[mask_id] - .5*stepsize * dKᵒ * ∇xᵒ[mask_id])
        elseif update == :rmmala_mom

             ndistr = MvNormal(zeros(d*n),stepsize*invK)
             xᵒ[mask_id] .= x[mask_id] .+ .5 * stepsize * inv_dK * ∇x[mask_id] .+  rand(ndistr)
             cfgᵒ = ForwardDiff.GradientConfig(uᵒ, xᵒ, ForwardDiff.Chunk{2*d*n}())
             ForwardDiff.gradient!(∇xᵒ, uᵒ, xᵒ, cfgᵒ) # Xᵒ gets overwritten but does not change
              accinit = sum(llᵒ) - sum(ll) -
                        logpdf(ndistr,xᵒ[mask_id] - x[mask_id] - .5*stepsize * inv_dK * ∇x[mask_id]) +
                       logpdf(ndistr,x[mask_id] - xᵒ[mask_id] - .5*stepsize * inv_dK * ∇xᵒ[mask_id])
       elseif update == :rmrw_mom

            ndistr = MvNormal(zeros(d*n),stepsize*invK)
            xᵒ[mask_id] .= x[mask_id]  .+  rand(ndistr)
            uᵒ(xᵒ)  # writes into llᵒ, don't need gradient info here
            accinit = sum(llᵒ) - sum(ll)  # proposal is symmetric
         end
         # add prior on momenta to accinit
        logpriorterm = logpdf(priormom, vcat(p(deepvec2state(xᵒ))...)) - logpdf(priormom, vcat(p(deepvec2state(x))...))
        # @show accinit
        # @show logpriorterm
        accinit += logpriorterm
        # MH acceptance decision
        if log(rand()) <= accinit
            #println("update initial state ", update, " accinit: ", round(accinit;digits=3), "  accepted")
            for k in 1:Q.nshapes
                for i in eachindex(X[1].yy)
                    X[k].yy[i] .= Xᵒ[k].yy[i]
                end
            end
            x .= xᵒ
            ∇x .= ∇xᵒ
            ll .= llᵒ
            accepted = 1.0
        else
            #println("update initial state ", update, " accinit: ", round(accinit;digits=3), "  rejected")
            accepted = 0.0
            xᵒ .= x  # for next call to update initial state these should be the same
            ∇xᵒ .= ∇x
        end
    end
    accepted
end


"""
    update_pars!(X, Xᵒ,W, Q, x, ll, priorθ, covθprop, obsinfo)

For fixed Wiener increments and initial state, update parameters by random-walk-MH.

## Writes into / modifies
`X`: landmark paths under landmarks process with parameter θ
`ll`: vector of loglikelihoods (one for each shape)

## Returns
`Q, accept`: where `accept` is 1/0 according to accept/reject in the MH-step.
"""
function update_pars!(X, ll, Xᵒ, W, Q , priorθ, covθprop, obsinfo)
    θ = getpars(Q)
    distr = MvLogNormal(MvNormal(log.(θ),covθprop))
    θᵒ = rand(distr)
    distrᵒ = MvLogNormal(MvNormal(log.(θᵒ),covθprop))
    Qᵒ = adjust_to_newpars(Q, θᵒ)
    update_guidrec!(Qᵒ, obsinfo)   # compute backwards recursion with parameter θᵒ
    x0 = X[1].yy[1]
    llᵒ = gp!(LeftRule(), Xᵒ,x0, W, Qᵒ; skip=sk)

    A = sum(llᵒ) - sum(ll) +
        logpdf(priorθ, θᵒ) - logpdf(priorθ, θ) +
        logpdf(distrᵒ, θ) - logpdf(distr, θᵒ)
    if log(rand()) <= A
        ll .= llᵒ
        for k in 1:Q.nshapes
            for i in eachindex(X[1].yy)
                X[k].yy[i] .= Xᵒ[k].yy[i]
            end
        end
        Q = Qᵒ
        accept = 1.0
    else
        accept = 0.0
    end
    Q, accept
end


######### adaptation of mcmc tuning parameters #################

"""
    adaptmalastep!(δ, n, accinfo, η, update; adaptskip = 15, targetaccept=0.5)

Adapt the step size for mala_mom updates.
The adaptation is multiplication/divsion by exp(η(n)).
"""
function adaptmalastep(δ, n, accinfo, η, update; adaptskip = 15, targetaccept=0.5)
    recent_mean = mean(accinfo[end-adaptskip+1:end])
    if recent_mean > targetaccept
        return δ *= exp(η(n))
    else
        return δ *= exp(-η(n))
    end
end

"""
"""
    adaptpcnstep(n, accpcn, ρ, nshapes, η; adaptskip = 15, targetaccept=0.5)

"""
    adaptparstep(n,accinfo,covθprop, η;  adaptskip = 15, targetaccept=0.5)

Adjust `covθprop-parameter` adaptively, every `adaptskip` steps.
For that the assumed multiplicative parameter is first tranformed to (-∞, ∞), then updated, then mapped back to (0,∞)
"""
function adaptparstep(covθprop, n, accinfo, η;  adaptskip = 15, targetaccept=0.5)
    recent_mean = mean(accinfo[end-adaptskip+1:end])
    if recent_mean > targetaccept
        return    covθprop *= exp(2*η(n))
    else
        return    covθprop *= exp(-2*η(n))
    end
end

"""
    sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))
"""
sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))

"""
    invsigmoid(z::Real) = log(z/(1-z))
"""
invsigmoid(z::Real) = log(z/(1-z))

"""
    adaptpcnstep(n, accpcn, ρ, nshapes, η; adaptskip = 15, targetaccept=0.5)

Adjust pcN-parameter `ρ` adaptive every `adaptskip` steps.
For that `ρ` is first tranformed to (-∞, ∞), then updated, then mapped back to (0,1)
"""
function adaptpcnstep(ρ, n, accinfo, η; adaptskip = 15, targetaccept=0.5)
    recentmean = mean(accinfo[end-adaptskip+1:end])
    if recentmean > targetaccept
        return sigmoid(invsigmoid(ρ) - η(n))
    else
        return sigmoid(invsigmoid(ρ) + η(n))
    end
end
