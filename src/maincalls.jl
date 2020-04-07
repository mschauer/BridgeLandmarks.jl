"""
landmarkmatching
"""

using Revise

using BridgeLandmarks
const BL = BridgeLandmarks
using RCall
using Random
using Distributions
using DataFrames
using DelimitedFiles
using CSV
using StaticArrays
using LinearAlgebra
using JLD2
using FileIO

Random.seed!(9)

workdir = @__DIR__
cd(workdir)
include(joinpath(BL.dir(),"scripts", "postprocessing.jl"))
outdir = workdir
mkpath(joinpath(outdir, "forward"))

if false # make example datasets with configurations
    dat = load("../experiments/exp1/data_exp1.jld2")
    xobs0 = dat["xobs0"]
    xobsT = dat["xobsT"]
    writedlm("landmarks0.txt", hcat(extractcomp(xobs0,1), extractcomp(xobs0,2)))
    writedlm("landmarksT.txt", hcat(extractcomp(xobsT,1), extractcomp(xobsT,2)))
end

# pass matrices to main function
landmarks0 = readdlm("landmarks0.txt")
landmarksT = readdlm("landmarksT.txt")






function landmarkmatching(landmarks0,landmarksT;
        model=:ms,
        updatescheme = [:innov, :mala_mom],
        σobs = 0.01,
        dt = 0.01,
        ρinit = 0.9,              # pcN-step
        covθprop = [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04],
        adaptskip = 20,  # adapt mcmc tuning pars every adaptskip iters
        maxnrpaths = 10, # update at most maxnrpaths Wiener increments at once
        ITER = 50,
        subsamples = 0:10:ITER
    )

    if model==:ms
        δinit = [0.001, 0.1] # first comp is not used
    else
        δinit = [0.1, 0.1] # first comp is not used
    end
    η(n) = min(0.2, 10/n)  # adaptation rate for adjusting tuning pars
    outdir=@__DIR__

    xobs0 = [PointF(r...) for r in eachrow(landmarks0)]
    xobsT = [[PointF(r...) for r in eachrow(landmarksT)]]

    @assert length(landmarks0)==length(landmarksT) "The two given landmark configurations do not have the same number of landmarks."
    nshapes = 1
    n = length(xobs0)
    Σobs = fill([σobs^2 * one(UncF) for i in 1:n],2) # noise on observations
    T = 1.0; dt = 0.01; t = 0.0:dt:T; tt_ =  tc(t,T)
    tp = tuningpars_mcmc(ρinit, maxnrpaths, δinit,covθprop, η, adaptskip)
    obs_atzero = true
    fixinitmomentato0 = false

    ################################# initialise P #################################
    ainit = mean(norm.([xobs0[i]-xobs0[i-1] for i in 2:n]))/2.0
    if model == :ms
        cinit = 0.2
        γinit = 2.0
        P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
    elseif model == :ahs
        cinit = 0.02
        γinit = 0.2
        stdev = 0.75
        nfsinit = construct_nfs(2.5, stdev, γinit)
        P = Landmarks(ainit, cinit, n, 2.5, stdev, nfsinit)
    end

    ################## prior specification with θ = (a, c, γ) ########################
    priorθ = product_distribution([Exponential(ainit), Exponential(cinit), Exponential(γinit)])
    κ = 100.0
    priormom = MvNormalCanon(zeros(d*n), gramkernel(xobs0,P)/κ)

    xinit = State(xobs0, zeros(PointF,P.n))
    mT = zeros(PointF, n)

print(typeof(xobs0))
print(typeof(xobsT))
print(typeof(Σobs))

if true
    start = time() # to compute elapsed time
        Xsave, parsave, objvals, accpcn, accinfo, δ, ρ, covθprop =
        lm_mcmc(tt_, (xobs0,xobsT), Σobs, mT, P,
                  obs_atzero, fixinitmomentato0, ITER, subsamples,
                  xinit, tp, priorθ, priormom, updatescheme,
                outdir)
    elapsed = time() - start

    #----------- post processing -------------------------------------------------
    println("Elapsed time: ",round(elapsed/60;digits=2), " minutes")
    perc_acc_pcn = mean(accpcn)*100
    println("Acceptance percentage pCN step: ", round(perc_acc_pcn;digits=2))
    write_mcmc_iterates(Xsave, tt_, n, nshapes, subsamples, outdir)
    write_info(model,ITER, n, tt_, updatescheme, Σobs, tp, ρ, δ, perc_acc_pcn, elapsed, outdir)
    write_observations(xobs0, xobsT, n, nshapes, outdir)
    write_acc(accinfo, accpcn, nshapes,outdir)
    write_params(parsave, 0:ITER, outdir)
    write_noisefields(P, outdir)
end
end

landmarkmatching(landmarks0, landmarksT)


using BridgeLandmarks
F = [(i==j) * one(UncF) for i in 1:5, j in 1:3]  # pick position indices
F
struct Test
    F
end

obj = Test(F)
obj
show(obj.F)
show(obj)
import Base.show
function show(io::IO, mime::MIME"text/plain",obj::Test)
    print(io,mime, obj.F)
end

function show(io::IO, obj::Test)
    show(io,obj.F)
end
show(obj)
