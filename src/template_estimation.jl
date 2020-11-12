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
        xinitq = nothing
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
- `xinitq::Array{PointF}`: Initialisation for the shape. If not provided xobsT[1] will be taken for initialisation.
- `printskip`: skip every printskip observations in writing output to console
## Example:
```
    # Make test example data set
    n = 5
    nshapes = 7
    q0 = [PointF(2.0cos(t), sin(t))  for t in collect(0:(2pi/n):2pi)[2:end]] # initial shape is an ellipse
    x0 = State(q0, randn(PointF,n))
    xobsT = Vector{PointF}[]

    T = 1.0; dt = 0.01; t = 0.0:dt:T
    Ptrue = MarslandShardlow(2.0, 0.1, 1.7, 0.0, n)
    for k in 1:nshapes
            Wf, Xf = BridgeLandmarks.landmarksforward(t, x0, Ptrue)
            push!(xobsT, Xf.yy[end].q + σobs * randn(PointF,n))
    end

    template_estimation(xobsT)
`"""
function template_estimation(
    xobsT;#::Array{Array{PointF}};
    pars = Pars_ms(),
    updatescheme = [:innov, :rmmala_pos, :parameter],
    ITER = 100,
    outdir=@__DIR__,
    Σobs = nothing, ainit = nothing, xinitq = nothing, printskip=20)

#    model = pars.model
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
    obsinfo = set_obsinfo(xobs0,xobsT,Σobs, obs_atzero,fixinitmomentato0)

    ################################# initialise P #################################
    if isnothing(ainit)
        ainit = 0.5*mean(norm.([xobsT[1][i]-xobsT[1][i-1] for i in 2:n]))
    end
    cinit = pars.cinit
    γinit = pars.γinit
    if pars.model == :ms
        P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
    elseif pars.model == :ahs
        nfsinit = construct_nfs(pars.db, pars.stdev, γinit)
        P = Landmarks(ainit, cinit, n, pars.db , pars.stdev, nfsinit)
    end

    ################## prior specification with θ = (a, γ) ########################
    priorθ = product_distribution([pars.aprior, pars.γprior])
    priormom = FlatPrior()

    mT = zeros(PointF, n)
    if isnothing(xinitq)
        xinitq = xobsT[1]
    end
    mT = zeros(PointF, n)
    xinit = State(xinitq, mT)

    start = time()
          Xsave, parsave, initendstates_save, accinfo, δ, ρ, δa =
                lm_mcmc(tt, obsinfo, mT, P, ITER, subsamples, xinit, pars, priorθ, priormom, updatescheme, outdir, printskip)
    elapsed = time() - start

    write_output(obsinfo.xobs0, obsinfo.xobsT, parsave, Xsave, initendstates_save, elapsed, accinfo, tt, n,nshapes,subsamples,ITER, updatescheme, Σobs, pars, ρ, δ, P, outdir)
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

    @warn "Not fully tested. Please provide landmark configurations as an array with each element of type Vector{PointF}"
    # convert landmark coordinates to arrays of PointF
    xobsT = [ [PointF(r...) for r in eachrow(lmT)]  for lmT in eachindex(landmarksT)]
    template_estimation(xobsT; pars=pars,updatescheme=updatescheme,
        ITER=ITER, outdir=outdir, Σobs=Σobs, ainit=ainit)
    nothing
end
