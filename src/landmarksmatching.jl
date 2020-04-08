include(joinpath(BridgeLandmarks.dir(),"scripts", "postprocessing.jl"))

"""
landmarksmatching(
        landmarks0::Matrix{Float64},landmarksT::Matrix{Float64};
        pars = Pars_ms(),
        updatescheme = [:innov, :mala_mom],
        ITER = 50,
        outdir=@__DIR__,
        Σobs = nothing,
        ainit = nothing
    )

landmarks0: n x d matrix of landmarks in Rd at time 0
landmarksT: n x d matrix of landmarks in Rd at time T
pars: either pars_ms() or pars_ahs() (this selects the model and default parameter values)
updatescheme: array of mcmc updates
ITER: number of iterations
outdir: path of directory to which output is written
Σobs: If provided, a length two array, where the first (second) element gives an Array{Unc} elements, where each element specifies
    the covariance of the extrinsic noise for each landmark at time 0 (final time T). If not provided, Σobs is constructed using
    the value of pars.σobs
anit: Hamiltonian kernel parameter. If not provided, defaults to setting
    ainit = mean(norm.([xobs0[i]-xobs0[i-1] for i in 2:n]))
"""
function landmarksmatching(
    landmarks0::Matrix{Float64},landmarksT::Matrix{Float64};
    pars = Pars_ms(),
    updatescheme = [:innov, :mala_mom],
    ITER = 100,
    outdir=@__DIR__,
    Σobs = nothing,
    ainit = nothing
    )

    model = pars.model
    # convert landmark coordinates to arrays of PointF
    xobs0 = [PointF(r...) for r in eachrow(landmarks0)]
    xobsT = [[PointF(r...) for r in eachrow(landmarksT)]]
    n = length(xobs0)
    nshapes = 1
    @assert length(landmarks0)==length(landmarksT) "The two given landmark configurations do not have the same number of landmarks."

    if isnothing(Σobs)
        Σobs = fill([pars.σobs^2 * one(UncF) for i in 1:n],2) # noise on observations
    end
    T = 1.0; t = 0.0:pars.dt:T; tt = tc(t,T)
    obs_atzero = true
    fixinitmomentato0 = false
    subsamples = 0:pars.skip_saveITER:ITER

    ################################# initialise P #################################
    if isnothing(ainit)
        ainit = mean(norm.([xobs0[i]-xobs0[i-1] for i in 2:n]))
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
    priormom = MvNormalCanon(zeros(d*n), gramkernel(xobs0,P)/pars.κ)

    xinit = State(xobs0, zeros(PointF,P.n))
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
    write_observations(xobs0, xobsT, n, nshapes, outdir)
    write_acc(accinfo, accpcn, nshapes,outdir)
    write_params(parsave, 0:ITER, outdir)
    write_noisefields(P, outdir)
    nothing
end
