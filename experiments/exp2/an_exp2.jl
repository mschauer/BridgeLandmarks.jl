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
dat = load("data_exp2.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
nshapes = dat["nshapes"]
q0 = dat["q0"]

################################# start settings #################################
ITER = 2_000
subsamples = 0:10:ITER

model = [:ms, :ahs][1]
fixinitmomentato0 = true
obs_atzero = false
updatescheme =  [:innov, :rmmala_pos, :parameter]

σobs = 0.01   # noise on observations
Σobs = fill([σobs^2 * one(UncF) for i in 1:n],2)

T = 1.0; dt = 0.01; t = 0.0:dt:T; tt_ =  tc(t,T)

################################# MCMC tuning pars #################################
ρinit = 0.7              # pcN-step
covθprop = [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04]
if model==:ms
    δinit = [0.002, 0.1]
else
    δinit = [0.01, 0.1]
end
η(n) = min(0.2, 10/n)  # adaptation rate for adjusting tuning pars
adaptskip = 20  # adapt mcmc tuning pars every adaptskip iters
maxnrpaths = 10 # update at most maxnrpaths Wiener increments at once
tp = MCMC_TuningPars(ρinit, maxnrpaths, δinit,covθprop,η,adaptskip)

################################# initialise P #################################
ainit = mean(norm.([xobsT[1][i]-xobsT[1][i-1] for i in 2:n]))/2.0
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
struct FlatPrior end
import Distributions.logpdf
logpdf(::FlatPrior, _x) = 0.0
priormom = FlatPrior()

mT = zeros(PointF,n)
# deliberately take wrong initial landmark configuration
xinitq = xobsT[1]
θ, ψ =  π/6, 0.25
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
xinitq_adj = [rot * stretch * xinitq[i] for i in 1:n ]
xinit = State(xinitq_adj, mT)

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
write_observations(q0, xobsT, n, nshapes, outdir)
write_acc(accinfo, accpcn, nshapes,outdir)
write_params(parsave, 0:ITER, outdir)
write_noisefields(P, outdir)
