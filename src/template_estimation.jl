#include(joinpath(BridgeLandmarks.dir(),"scripts", "postprocessing.jl"))


"""
    template_estimation(
        xobsT::Array{PointF};
        pars = Pars_ms(),
        updatescheme = [:innov, :rmmala_pos, :parameter],
        ITER = 100,
        outdir=@__DIR__,
        Σobs = nothing,
        ainit = nothing
    )

## Arguments
- `xobsT`: an array, where each element of the array gives the coordinates of a shape (i.e. the landmarks, which
are represented as an array of PointF)

## Optional arguments
- `pars`: either `pars_ms()`` or `pars_ahs()`` (this selects the model and default parameter values)
- `updatescheme`: array of mcmc updates
- `ITER`: number of iterations
- `outdir`: path of directory to which output is written
- `Σobs`: If provided, a length two array, where the first (second) element gives an Array{Unc} elements, where each element specifies
    the covariance of the extrinsic noise for each landmark at time 0 (final time T). If not provided, Σobs is constructed using
    the value of pars.σobs
- `anit`: Hamiltonian kernel parameter. If not provided, defaults to setting
    `ainit = mean(norm.([xobs0[i]-xobs0[i-1] for i in 2:n]))`

## Example:
```
    dat = load("../experiments/exp2/data_exp2.jld2")
    xobsT = dat["xobsT"]

    template_estimation(xobsT)
`"""
function template_estimation(
    xobsT;#::Array{Array{PointF}};
    pars = Pars_ms(),
    updatescheme = [:innov, :rmmala_pos, :parameter],
    ITER = 10,
    outdir=@__DIR__,
    Σobs = nothing,
    ainit = nothing
    )

    model = pars.model
    xobs0 = [] # as it is not known

    n = length(xobsT[1])
    nshapes = length(xobsT)

    if isnothing(Σobs)
        Σobs = fill([pars.σobs^2 * one(UncF) for i in 1:n],2) # noise on observations
    end
    T = 1.0; t = 0.0:pars.dt:T; tt = tc(t,T)
    obs_atzero = false
    fixinitmomentato0 = true
    subsamples = 0:pars.skip_saveITER:ITER

    ################################# initialise P #################################
    if isnothing(ainit)
        ainit = mean(norm.([xobsT[1][i]-xobsT[1][i-1] for i in 2:n]))
    end
    cinit = pars.cinit
    γinit = pars.γinit
    if model == :ms
        P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
    elseif model == :ahs
        nfsinit = construct_nfs(pars.db, pars.stdev, γinit)
        P = Landmarks(ainit, cinit, n, pars.db , pars.stdev, nfsinit)
    end

    ################## prior specification with θ = (a, c, γ) ########################
    priorθ = product_distribution([Exponential(ainit), Exponential(cinit), Exponential(γinit)])
    priormom = FlatPrior()

    mT = zeros(PointF, n)

    xinitq = xobsT[1]
    θ, ψ =  π/6, 0.25
    rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
    stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
    xinitq_adj = [rot * stretch * xinitq[i] for i in 1:n ]
    xinit = State(xinitq_adj, mT)

    mT = zeros(PointF, n)
    start = time()
        Xsave, parsave, objvals, accpcn, accinfo, δ  , ρ, covθprop =
                lm_mcmc(tt, (xobs0,xobsT), Σobs, mT, P,obs_atzero, fixinitmomentato0, ITER, subsamples,
                                                    xinit, pars, priorθ, priormom, updatescheme, outdir)
    elapsed = time() - start

    ################## post processing ##################
    println("Elapsed time: ",round(elapsed/60;digits=2), " minutes")
    perc_acc_pcn = mean(accpcn)*100
    println("Acceptance percentage pCN step: ", round(perc_acc_pcn;digits=2))
    write_mcmc_iterates(Xsave, tt, n, nshapes, subsamples, outdir)
    write_info(model,ITER, n, tt, updatescheme, Σobs, pars, ρ, δ , perc_acc_pcn, elapsed, outdir)
    q0 = 0.0 * xobsT[1]
    write_observations(q0, xobsT, n, nshapes, outdir)
    write_acc(accinfo, accpcn, nshapes,outdir)
    write_params(parsave, 0:ITER, outdir)
    write_noisefields(P, outdir)
    nothing
end

"""
    template_estimation(
        landmarksT::Array{Matrix{Float64}};
        pars = Pars_ms(),
        updatescheme = [:innov, :mala_mom],
        ITER = 100,
        outdir=@__DIR__,
        Σobs = nothing,
        ainit = nothing
    )
"""
function template_estimation(
    landmarksT::Array{Matrix{Float64}};
    pars = Pars_ms(),
    updatescheme = [:innov, :mala_mom],
    ITER = 100,
    outdir=@__DIR__,
    Σobs = nothing,
    ainit = nothing
    )

    # convert landmark coordinates to arrays of PointF
    xobsT = [ [PointF(r...) for r in eachrow(lmT)]  for lmT in eachindex(landmarksT)]
    template_estimation(xobsT; pars=pars,updatescheme=updatescheme,
        ITER=ITER, outdir=outdir, Σobs=Σobs, ainit=ainit)
    nothing
end
