using Revise

using BridgeLandmarks
const BL=BridgeLandmarks
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
dat = load("data_cardiac.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
nshapes = dat["nshapes"]

################################# start settings #################################
ITER = 2000
subsamples = 0:10:ITER

model = [:ms, :ahs][1]
fixinitmomentato0 = true
fixinitmomentato0 = true
obs_atzero = false
updatescheme =  [:innov, :rmmala_pos, :parameter]

σobs = 0.01   # noise on observations
Σobs = fill([σobs^2 * one(UncF) for i in 1:n],2)

T = 1.0; dt = 0.01; t = 0.0:dt:T; tt_ =  tc(t,T)

################################# MCMC tuning pars #################################
ρinit = 0.4              # pcN-step
covθprop =  0.001 *  Diagonal(fill(1.0,3))
if model==:ms
    δinit = [0.0002, 0.1]
else
    δinit = [0.01, 0.1]
end
η(n) = min(0.2, 10/n)  # adaptation rate for adjusting tuning pars
adaptskip = 20  # adapt mcmc tuning pars every adaptskip iters
maxnrpaths = 10 # update at most maxnrpaths Wiener increments at once
tp = tuningpars_mcmc(ρinit, maxnrpaths, δinit,covθprop,η,adaptskip)

################################# initialise P #################################
ainit = 0.1
if model == :ms
    cinit = 0.2
    γinit = 1.0
    P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
elseif model == :ahs
    cinit = 0.05
    γinit = 0.5
    stdev = 0.75
    nfsinit = construct_nfs(2.5, stdev, γinit)
    P = Landmarks(ainit, cinit, n, 2.5, stdev, nfsinit)
end

################## prior specification with θ = (a, c, γ) ########################
priorθ = product_distribution([Exponential(ainit), Exponential(cinit), Exponential(γinit)])
struct FlatPrior end
import Distributions.logpdf
logpdf(::FlatPrior, _x) = 0.0
priormom = FlatPrior()

#########################
mT = zeros(PointF,n)   # vector of momenta at time T used for constructing guiding term #mT = randn(PointF,P.n)
xinitq = xobsT[1]
xinit = State(xinitq, mT)

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
q0 = xinitq  # just arbitrary, not used, as q0 is not observed
write_observations(q0, xobsT, n, nshapes, outdir)
write_acc(accinfo, accpcn, nshapes,outdir)
write_params(parsave, 0:ITER, outdir)
write_noisefields(P, outdir)
