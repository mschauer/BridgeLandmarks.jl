#include(joinpath(BridgeLandmarks.dir(),"scripts", "postprocessing.jl"))

"""
    landmarksmatching(
        xobs0::Array{PointF},xobsT::Array{PointF};
        pars = Pars_ms(),
        updatescheme = [:innov, :mala_mom],
        ITER = 100,
        outdir=@__DIR__,
        Σobs = nothing,
        ainit = nothing
    )

## Arguments
- `xobs0`: Array of PointF (coordinates of inital shape)
- `xobsT`: Array of PointF (coordinats of shape at time T)

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
    # prepare some test data
    n = 7
    xobs0 = [PointF(2.0cos(t), sin(t))/4.0  for t in collect(0:(2pi/n):2pi)[2:end]]
    θ, ψ =  π/6, 0.1
    rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
    stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
    shift = [0.5, -1.0]
    xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:n]

    landmarksmatching(xobs0,xobsT; ITER=10)
    landmarksmatching(xobs0,xobsT; ITER=10, pars=BL.Pars_ahs())
```
"""
function landmarksmatching(
    xobs0::Array{PointF},xobsT::Array{PointF};
    pars = Pars_ms(),
    updatescheme = [:innov, :mala_mom],
    ITER = 100,
    outdir=@__DIR__,
    Σobs = nothing,
    ainit = nothing
    )

    model = pars.model
    n = length(xobs0)
    nshapes = 1
    @assert length(xobs0)==length(xobsT) "The two given landmark configurations do not have the same number of landmarks."

    if isnothing(Σobs)
        Σobs = fill([pars.σobs^2 * one(UncF) for i in 1:n],2) # noise on observations
    end
    T = 1.0; t = 0.0:pars.dt:T; tt = tc(t,T)
    obs_atzero = true
    fixinitmomentato0 = false
    subsamples = 0:pars.skip_saveITER:ITER
    obsinfo = set_obsinfo(xobs0,[xobsT],Σobs, obs_atzero,fixinitmomentato0)

    ################################# initialise P #################################
    if isnothing(ainit)
        ainit 0.5*mean(norm.(diff(xobs0)))
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
    priorθ = product_distribution([pars.aprior, pars.γprior])
    priormom = MvNormalCanon(zeros(d*n), gramkernel(xobs0,P)/pars.κ)

    xinit = State(xobs0, zeros(PointF,P.n))
    mT = zeros(PointF, n)
    start = time()
          Xsave, parsave, accinfo, δ, ρ, δa =
                lm_mcmc(tt, obsinfo, mT, P, ITER, subsamples, xinit, pars, priorθ, priormom, updatescheme, outdir)
    elapsed = time() - start

    write_output(obsinfo.xobs0, obsinfo.xobsT, parsave, Xsave, elapsed, accinfo, tt, n,nshapes,subsamples,ITER, updatescheme, Σobs, pars, ρ, δ, P, outdir)
    nothing
end

"""
    landmarksmatching(
        landmarks0::Matrix{Float64},landmarksT::Matrix{Float64};
        pars = Pars_ms(),
        updatescheme = [:innov, :mala_mom],
        ITER = 100,
        outdir=@__DIR__,
        Σobs = nothing,
        ainit = nothing
    )

landmarksmatching, where the input does not consist of arrays of points, but both the initial shape and final shape
are represented by a matrix. Each row of the matrix gives the coordinates of a landmark.
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

    @assert size(landmarks0)==size(landmarksT)  "landmarks0 and landmarksT should have the same dimensions."
    # convert landmark coordinates to arrays of PointF
    xobs0 = [PointF(r...) for r in eachrow(landmarks0)]
    xobsT = [PointF(r...) for r in eachrow(landmarksT)]
    landmarksmatching(xobs0, xobsT; pars=pars,updatescheme=updatescheme,
        ITER=ITER, outdir=outdir, Σobs=Σobs, ainit=ainit)
    nothing
end
