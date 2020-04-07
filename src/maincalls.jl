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
        pars = Pars(),
        ITER = 50,
        subsamples = 0:10:ITER,
        outdir=@__DIR__
    )

    xobs0 = [PointF(r...) for r in eachrow(landmarks0)]
    xobsT = [[PointF(r...) for r in eachrow(landmarksT)]]

    @assert length(landmarks0)==length(landmarksT) "The two given landmark configurations do not have the same number of landmarks."
    nshapes = 1
    n = length(xobs0)
    Σobs = fill([pars.σobs^2 * one(UncF) for i in 1:n],2) # noise on observations
    T = 1.0; t = 0.0:pars.dt:T; tt_ =  tc(t,T)
    obs_atzero = true
    fixinitmomentato0 = false

    ################################# initialise P #################################

    ainit = mean(norm.([xobs0[i]-xobs0[i-1] for i in 2:n]))
    if model == :ms
        δinit = δinit_ms
        cinit = pars.cinit_ms #0.2
        γinit = pars.γinit_ms #2.0
        P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
    elseif model == :ahs
        δinit = δinit_ahs
        cinit = pars.cinit_ahs 0.02
        γinit = pars.γinit_ahs 0.2
        stdev = pars.stdev_ahs 0.75
        nfsinit = construct_nfs(2.5, stdev, γinit)
        P = Landmarks(ainit, cinit, n, pars.db_ahs , stdev, nfsinit)  # 2.5
    end

    ################## prior specification with θ = (a, c, γ) ########################
    priorθ = product_distribution([Exponential(ainit), Exponential(cinit), Exponential(γinit)])
    priormom = MvNormalCanon(zeros(d*n), gramkernel(xobs0,P)/pars.κ)

    xinit = State(xobs0, zeros(PointF,P.n))
    mT = zeros(PointF, n)

    start = time()
        Xsave, parsave, objvals, accpcn, accinfo, δ, ρ, covθprop =
        lm_mcmc(tt_, (xobs0,xobsT), Σobs, mT, P,
                  obs_atzero, fixinitmomentato0, ITER, subsamples,
                  xinit, pars, priorθ, priormom, updatescheme,
                outdir)
    elapsed = time() - start


if false
    #----------- post processing -------------------------------------------------
    println("Elapsed time: ",round(elapsed/60;digits=2), " minutes")
    perc_acc_pcn = mean(accpcn)*100
    println("Acceptance percentage pCN step: ", round(perc_acc_pcn;digits=2))
    write_mcmc_iterates(Xsave, tt_, n, nshapes, subsamples, outdir)
    write_info(model,ITER, n, tt_, updatescheme, Σobs, pars, ρ, δ, perc_acc_pcn, elapsed, outdir)
    write_observations(xobs0, xobsT, n, nshapes, outdir)
    write_acc(accinfo, accpcn, nshapes,outdir)
    write_params(parsave, 0:ITER, outdir)
    write_noisefields(P, outdir)
end
end

# @enter landmarkmatching(landmarks0, landmarksT)
# Juno.@enter landmarkmatching(landmarks0, landmarksT)

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
