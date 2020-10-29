# ellipse to rotated and shifted ellipse
# initial and final landmark positions observed

# just as exp1, but only larger noise on landmarks 9--18,
# and also updating of the initial positions
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

#-------- read data ----------------------------------------------------------
dat = load("data_exp1.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
nshapes = dat["nshapes"]

################################# start settings #################################
ITER = 500
subsamples = 0:10:ITER

model = [:ms, :ahs][1]
fixinitmomentato0 = false
obs_atzero = true
updatescheme =  [:innov, :mala_mom]

# noise on observations
σobs0 = 0.01
Σobs0 = [σobs0^2 * one(UncF) for i in 1:n]
σobsT = vcat(fill(σobs,9), fill(0.1,n-9))
ΣobsT = [σobsT[i]^2 * one(UncF) for i in 1:n]
Σobs = [Σobs0, ΣobsT]

T = 1.0; dt = 0.01; t = 0.0:dt:T; tt_ =  tc(t,T)

################################# MCMC tuning pars #################################
ρinit = 0.9              # pcN-step
covθprop = [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04]
if model==:ms
    δinit = [0.0001, 0.1] # first comp is not used
else
    δinit = [0.001, 0.1] # first comp is not used
end
η(n) = min(0.2, 10/n)  # adaptation rate for adjusting tuning pars
adaptskip = 20  # adapt mcmc tuning pars every adaptskip iters
maxnrpaths = 10 # update at most maxnrpaths Wiener increments at once
tp = tuningpars_mcmc(ρinit, maxnrpaths, δinit,covθprop,η,adaptskip)

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

#########################
xobsT = [xobsT]
xinit = State(xobs0, zeros(PointF,P.n))
mT = zeros(PointF, n)

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
write_observations(xobs0, xobsT, n, nshapes, x0,outdir)
write_acc(accinfo, accpcn, nshapes,outdir)
write_params(parsave, 0:ITER, outdir)
write_noisefields(P, outdir)
