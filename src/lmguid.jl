import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

"""
    lm_mcmc(t, (xobs0,xobsT), Σobs, mT, P,
          obs_atzero, fixinitmomentato0, ITER, subsamples,
          xinit, pars, priorθ, priormom, updatescheme,
          outdir)

This is the main function call for doing either MCMC or SGD for landmark models
Backward ODEs used are in terms of the LMμ-parametrisation

## Arguments
- `t`:      time grid
- `(xobs0,xobsT)`: observations at times 0 and T
- `Σobs`: array with covariance matrix of Gaussian noise assumed on each element of xobs0 and xobsT
- `mT`: vector of momenta at time T used for constructing guiding term
- `P`: target process
- `obs_atzero`: Boolean, if true there is an observation at time zero
- `fixinitmomentato0`: Boolean, if true we assume at time zero we observe zero momenta
- `ITER`: number of iterations
- `subsamples`: vector of indices of iterations that are to be saved
- `xinit`: initial guess on starting state
- `pars`: tuning pars for updates
- `priorθ`: prior on θ=(a,c,γ)
- `priormom`: prior on initial momenta
- `updatescheme`: specify type of updates
- `outdir` output directory for animation

## Returns:
- `Xsave`: saved iterations of all states at all times in tt_
- `parsave`: saved iterations of all parameter updates ,
- `objvals`: saved values of stochastic approximation to loglikelihood
- `accpcn`: acceptance percentages for pCN step
- `accinfo`: acceptance percentages for remaining mcmc-updates
- `δ`: value of `(δpos, δmom)` af the final iteration of the algorithm (these are stepsize parameters for updating initial positions and momenta respectively)
- `ρ`: value of `ρinit`  af the final iteration of the algorithm
- `covθprop`: value of `covθprop` at the final iteration of the algorithm
"""
function lm_mcmc(t, (xobs0,xobsT), Σobs, mT, P,
          obs_atzero, fixinitmomentato0, ITER, subsamples,
          xinit, pars, priorθ, priormom, updatescheme,
          outdir)

    lt = length(t)
    StateW = PointF
    dwiener = dimwiener(P)

    obsinfo = set_obsinfo(xobs0,xobsT,Σobs, obs_atzero,fixinitmomentato0)

    # initialise GuidedProposal!, which contains all info for simulating guided proposals
    nshapes = obsinfo.nshapes
    guidrec = [init_guidrec(t,obsinfo) for _ in 1:nshapes]  # memory allocation for backward recursion for each shape
    Paux = [auxiliary(P, State(xobsT[k],mT)) for k in 1:nshapes] # auxiliary process for each shape
    Q = GuidedProposal!(P,Paux,t,obsinfo.xobs0,obsinfo.xobsT,guidrec,nshapes,[mT for _ in 1:nshapes])
    update_guidrec!(Q, obsinfo)   # compute backwards recursion

    # initialise Wiener increments and forward simulate guided proposals
    X = [initSamplePath(t, xinit) for _ in 1:nshapes]
    W = [initSamplePath(t,  zeros(StateW, dwiener)) for _ in 1:nshapes]
    for k in 1:nshapes
        sample!(W[k], Wiener{Vector{StateW}}())
    end
    x = deepvec(xinit)
    ∇x = deepcopy(x)

    # memory allocations, actual state at each iteration is (X,W,Q,x,∇x) (x, ∇x are initial state and its gradient)
    Xᵒ = deepcopy(X)
    Qᵒ = deepcopy(Q)
    Wᵒ = initSamplePath(t,  zeros(StateW, dwiener))
    Wnew = initSamplePath(t,  zeros(StateW, dwiener))
    xᵒ = deepcopy(x)
    ∇xᵒ = deepcopy(x)


    ll = gp!(LeftRule(), X, xinit, W, Q; skip=sk)




    # setup containers for saving objects
    objvals = Float64[]             # keep track of (sgd approximation of the) loglikelihood
    Xsave = typeof(zeros(length(t) * P.n * 2 * d * nshapes))[]
    parsave = Vector{Float64}[]
    push!(Xsave, convert_samplepath(X))
    obj = sum(ll)
    push!(objvals, obj)
    push!(parsave, getpars(Q))

    accinfo = []                        # keeps track of accepted parameter and initial state updates
    accpcn = Int64[]                      # keeps track of nr of accepted pCN updates

    δ = [pars.δpos, pars.δmom]
    ρ = pars.ρinit
    covθprop = pars.covθprop

    for i in 1:ITER
        println();  println("iteration $i")

        for update in updatescheme
            if update == :innov
                #@timeit to "path update"
                accpcn = update_path!(X, Xᵒ, W, Wᵒ, Wnew, ll, x, Q, ρ, accpcn, pars.maxnrpaths)
                ρ = adaptpcnstep(i, accpcn, ρ, Q.nshapes, pars.η; adaptskip = pars.adaptskip)
            elseif update in [:mala_mom, :rmmala_mom, :rmrw_mom]
                #@timeit to "update mom"
                 obj, accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,:mcmc, Q, δ, update,priormom)
                push!(accinfo, accinfo_)
                δ[2] = adaptmalastep(i,accinfo,δ[2], pars.η, update; adaptskip = pars.adaptskip)
            elseif update in [:mala_pos, :rmmala_pos]
                #@timeit to "update pos"
                obj, accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,:mcmc, Q, δ, update,priormom)
                push!(accinfo, accinfo_)
                δ[1] = adaptmalastep(i,accinfo,δ[1], pars.η, update; adaptskip = pars.adaptskip)
            elseif update == :parameter
                #@timeit to "update par"
                accinfo_ = update_pars!(obsinfo,X, Xᵒ,W, Q, Qᵒ, x, ll, priorθ, covθprop)
                push!(accinfo, accinfo_)
                covθprop = adaptparstep(i,accinfo,covθprop, pars.η;  adaptskip = pars.adaptskip)
            elseif update == :sgd
                obj, accinfo_ = update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,:sgd, Q, δ, update,priormom)
            end
        end

        # adjust mT (the momenta at time T used in the construction of the guided proposal)
        if mod(i,pars.adaptskip)==0 && i < 100
            mTvec = [X[k][lt][2].p  for k in 1:nshapes]     # extract momenta at time T for each shape
            update_mT!(Q, mTvec, obsinfo)
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


        println("ρ ", ρ , ",   δ ", δ, ",   covθprop ", covθprop)


        # if write_at_half # write state halfway of lastindex
        #     indexhalf = searchsortedfirst(t, t[end]/2.0)
        #     save("forwardhalfway.jld", "Xhalf",X[1].yy[indexhalf])
        # end
    end

    Xsave, parsave, objvals, accpcn, accinfo, δ, ρ, covθprop
end










"""
    update_path!(X,Xᵒ,W,Wᵒ,Wnew,ll,x, Q, ρ, accpcn, maxnrpaths)

Update bridges for all shapes using Crank-Nicholsen scheme with parameter `ρ` (only in case the method is mcmc).
Newly accepted bridges are written into `(X,W)`, loglikelihood on each segment is written into vector `ll`
"""
function update_path!(X,Xᵒ,W,Wᵒ,Wnew,ll,x, Q, ρ, accpcn, maxnrpaths)
    x0 = deepvec2state(x)
    dw = dimwiener(Q.target)
    # From current state (x,W) with loglikelihood ll, update to (x, Wᵒ)
    for k in 1:Q.nshapes
        sample!(Wnew, Wiener{Vector{PointF}}())
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

        llᵒ_ = gp!(LeftRule(), Xᵒ[k], x0, Wᵒ, Q,k;skip=sk)
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
function slogρ!(x0deepv, Q, W,X,priormom,llout)
    x0 = deepvec2state(x0deepv)
    lltemp = gp!(LeftRule(), X, x0, W, Q; skip=sk)   #overwrites X
    llout .= ForwardDiff.value.(lltemp)
    sum(lltemp) + logpdf(priormom, vcat(p(x0)...))
end
slogρ!(Q, W, X,priormom, llout) = (x) -> slogρ!(x, Q, W,X,priormom,llout)


"""
    update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,sampler, Q::GuidedProposal!, δ, update,priormom)

## Arguments
- `X`:  current iterate of vector of sample paths
- `Xᵒ`: vector of sample paths to write proposal into
- `W`:  current vector of Wiener increments
- `ll`: current value of loglikelihood
- `x`, `xᵒ`, `∇x`, `∇xᵒ`: allocated vectors for initial state and its gradient (both actual and proposed values)
- `sampler`: either sgd (not checked yet) or mcmc
- `Q`: GuidedProposal!
- `δ`: vector with MALA stepsize for initial state positions (δ[1]) and initial state momenta (δ[2])
- `update`:  can be `:mala_pos`, `:mala_mom`, :rmmala_pos`, `:rmmala_mom`, ``:rmrw_mom`

## Writes into
- `X`: landmarks paths
- `x`: initial state of the landmark paths
- ``∇x`: gradient of loglikelihood with respect to initial state `x`
- `ll`: a vector with as k-th element the loglikelihood for the k-th shape

## Returns
- `obj`: loglikelihood (summed over all shapes) + logdensity of prior on initial momenta
- `(kernel = update, acc = accepted)` tuple with name of the executed update and 0/1 indicator (reject/accept)

"""
function update_initialstate!(X,Xᵒ,W,ll,x,xᵒ,∇x, ∇xᵒ,
                sampler, Q::GuidedProposal!, δ, update,priormom)
    P = Q.target
    n = P.n
    x0 = deepvec2state(x)
    llout = copy(ll)
    lloutᵒ = copy(ll)
    u = slogρ!(Q, W, X, priormom,llout)
    uᵒ = slogρ!(Q, W, Xᵒ, priormom,lloutᵒ)
    cfg = ForwardDiff.GradientConfig(u, x, ForwardDiff.Chunk{2*d*n}()) # 2*d*P.n is maximal

    if sampler ==:sgd  # CHECK VALIDITY LATER
        @warn "Option :sgd has not been checked carefully so far."
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
    update_pars!(obs_info,X, Xᵒ,W, Q, Qᵒ, x, ll, priorθ, covθprop)

For fixed Wiener increments and initial state, update parameters by random-walk-MH.

## Writes into
`X`: landmark paths under landmarks process with parameter θ
`Q`: guidrec and target-fields are updated according to new parameter θ

## Returns
`(kernel = ":parameterupdate", acc = accept)`: where `accept` is 1/0 according to accept/reject in the MH-step.
"""
function update_pars!(obs_info,X, Xᵒ,W, Q, Qᵒ, x, ll, priorθ, covθprop)
    θ = getpars(Q)
    distr = MvLogNormal(MvNormal(log.(θ),covθprop))
    θᵒ = rand(distr)
    distrᵒ = MvLogNormal(MvNormal(log.(θᵒ),covθprop))
    putpars!(Qᵒ,θᵒ)
    update_guidrec!(Qᵒ, obs_info)   # compute backwards recursion
    llᵒ = gp!(LeftRule(), Xᵒ, deepvec2state(x), W, Qᵒ; skip=sk)

    A = sum(llᵒ) - sum(ll) +
        logpdf(priorθ, θᵒ) - logpdf(priorθ, θ) +
        logpdf(distrᵒ, θ) - logpdf(distr, θᵒ)
    if log(rand()) <= A
        ll .= llᵒ
        # deepcopyto!(Q.guidrec,Qᵒ.guidrec)
        # Q.target = Qᵒ.target
        # deepcopyto!(Q.aux,Qᵒ.aux)
        deepcopyto!(X,Xᵒ)
        Q = Qᵒ
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
"""
    adaptpcnstep(n, accpcn, ρ, nshapes, η; adaptskip = 15, targetaccept=0.5)

"""
    adaptparstep(n,accinfo,covθprop, η;  adaptskip = 15, targetaccept=0.5)

Adjust `covθprop-parameter` adaptively, every `adaptskip` steps.
For that the assumed multiplicative parameter is first tranformed to (-∞, ∞), then updated, then mapped back to (0,∞)
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
