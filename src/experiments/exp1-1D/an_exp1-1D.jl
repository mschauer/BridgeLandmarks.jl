# ellipse to rotated and shifted ellipse
# initial and final landmark positions observed
using Revise

using BridgeLandmarks
const BL=BridgeLandmarks
using RCall
using Random
using Distributions
using DataFrames
using DelimitedFiles
using CSV
using StaticArrays
using LinearAlgebra
using JLD

Random.seed!(9)

workdir = @__DIR__
cd(workdir)
include(dirname(dirname(workdir))*"/postprocessing.jl")
outdir = workdir*("/")

#-------- read data ----------------------------------------------------------
dat = load("data_exp1-1D.jld")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
x0 = dat["x0"]
nshapes = dat["nshapes"]


################################# start settings #################################
ITER = 100#0v
subsamples = 0:10:ITER

model = [:ms, :ahs][1]
fixinitmomentato0 = false
obs_atzero = true
updatescheme =  [:innov, :mala_mom]#, :parameter] # for pars: include :parameter

if model==:ms
    σobs = 0.001   # noise on observations
    Σobs = [σobs^2 * one(UncF) for i in 1:n]
else
    σobs = 0.001   # noise on observations
    Σobs = [σobs^2 * one(UncF) for i in 1:n]
end

T = 1.0; dt = 0.001; t = 0.0:dt:T; tt_ =  tc(t,T)


################################# MCMC tuning pars #################################
ρinit = 0.9              # pcN-step
covθprop =   [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04]
if model==:ms
    δinit = [0.001, 0.1] # first comp is not used
else
    δinit = [0.1, 0.5] # first comp is not used
end
η(n) = min(0.2, 10/n)  # adaptation rate for adjusting tuning pars
adaptskip = 20  # adapt mcmc tuning pars every adaptskip iters
maxnrpaths = 10 # update at most maxnrpaths Wiener increments at once
tp = tuningpars_mcmc(ρinit, maxnrpaths, δinit,covθprop,η,adaptskip)

################################# initialise P #################################
ainit = mean(norm.([x0.q[i]-x0.q[i-1] for i in 2:n]))/2.0   # Let op: door 2 gedeeld
if model == :ms
    cinit = 0.2
    γinit = 2.0
    P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
elseif model == :ahs
    cinit = 0.02*10
    γinit = 0.2*10
    stdev = 0.75
    nfsinit = construct_nfs(9.0, stdev, γinit)
    P = Landmarks(ainit, cinit, n, 2.5, stdev, nfsinit)
end

################## prior specification with θ = (a, c, γ) ########################
#priorθ = product_distribution(fill(Exponential(1.0),3))
priorθ = product_distribution([Exponential(ainit), Exponential(cinit), Exponential(γinit)])
κ = 100.0
prior_momenta = MvNormalCanon(gramkernel(xobs0,P)/κ)
prior_positions = MvNormal(vcat(xobs0...), σobs)
logpriormom(x0) = logpdf(prior_momenta, vcat(BL.p(x0)...))# +logpdf(prior_positions, vcat(BL.q(x0)...))

#########################
xobsT = [xobsT]
#xinit = State(xobs0, zeros(PointF,P.n))
xinitp = [PointF(12.0), PointF(-15.0), PointF(-15.0)]
xinit = State(xobs0, xinitp)
mT = zeros(PointF,n)#xinitp#

start = time() # to compute elapsed time
    Xsave, parsave, objvals, accpcn, accinfo, δ, ρ, covθprop =
    lm_mcmc(tt_, (xobs0,xobsT), Σobs, mT, P,
              obs_atzero, fixinitmomentato0, ITER, subsamples,
              xinit, tp, priorθ, logpriormom, updatescheme,
            outdir)
elapsed = time() - start

#----------- post processing -------------------------------------------------
println("Elapsed time: ",round(elapsed/60;digits=2), " minutes")
perc_acc_pcn = mean(accpcn)*100
println("Acceptance percentage pCN step: ", round(perc_acc_pcn;digits=2))
write_mcmc_iterates(Xsave, tt_, n, nshapes, subsamples, outdir)
write_info(model,ITER, n, tt_, updatescheme, Σobs, tp, ρ, δ, perc_acc_pcn, elapsed, outdir)
write_observations(xobs0, xobsT, n, nshapes, x0,outdir)
write_acc(accinfo,accpcn,nshapes,outdir)
write_params(parsave,0:ITER,outdir)
write_noisefields(P,outdir)