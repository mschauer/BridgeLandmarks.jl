using Revise

using BridgeLandmarks
const BL=BridgeLandmarks
using RCall
using Plots
using Random
using Distributions
using DataFrames
using DelimitedFiles
using CSV
using StaticArrays
using LinearAlgebra
using JLD
using StaticArrays
using LinearAlgebra

Random.seed!(3)

workdir = @__DIR__
cd(workdir)
include(dirname(dirname(workdir))*"/postprocessing.jl")
outdir = workdir*("/")

model = [:ms, :ahs][1]
T = 1.0; dt = 0.005; t = 0.0:dt:T; tt_ =  tc(t,T)

n = 10
nshapes = 1

q0 = [PointF(2.0cos(t), sin(t))  for t in collect(0:(2pi/n):2pi)[2:end]]
fac = model==:ms ? 1.0 : 45.0
p0 = fac*[PointF(1.0, -3.0) for i in 1:n]
#p0 = zeros(PointF,n)
x0 = State(q0, p0)


ainit = mean(norm.([x0.q[i]-x0.q[i-1] for i in 2:n]))/2.0   # Let op: door 2 gedeeld
if model == :ms
    cinit = 0.2
    γinit = .2
    P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
elseif model == :ahs
    cinit = 0.02
    γinit = 0.2
    stdev = 0.75
    nfsinit = construct_nfs(2.5, stdev, γinit)
    P = Landmarks(ainit, cinit, n, 2.5, stdev, nfsinit)
end


# simulate forward
Wf, Xf = landmarksforward(tt_, x0, P)
xobsT =[ BL.q(Xf.yy[end])] # [Xf.yy[end].q[i] for i in 1:n ]

Xfsave = []# typeof(zeros(length(tt_) * P.n * 2 * d * nshapes))[]
push!(Xfsave, BL.convert_samplepath(Xf))
write_mcmc_iterates(Xfsave, tt_, n, nshapes, [1], outdir*"forward/")

#--------- MCMC tuning pars ---------------------------------------------------------



################################# start settings #################################
ITER = 200
subsamples = 0:1:ITER

fixinitmomentato0 = false
obs_atzero = true
updatescheme =  [:innov, :mala_mom]#, :parameter] # for pars: include :parameter

σobs = 0.01   # noise on observations
Σobs = [σobs^2 * one(UncF) for i in 1:n]

################################# MCMC tuning pars #################################
ρinit = 0.9              # pcN-step
covθprop =   [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04]
if model==:ms
    δinit = [0.001, 0.02] # first comp is not used
else
    δinit = [0.1, 0.1] # first comp is not used
end
η(n) = min(0.2, 10/n)  # adaptation rate for adjusting tuning pars
adaptskip = 20  # adapt mcmc tuning pars every adaptskip iters
maxnrpaths = 10 # update at most maxnrpaths Wiener increments at once
tp = tuningpars_mcmc(ρinit, maxnrpaths, δinit,covθprop,η,adaptskip)

################################# initialise P #################################
xobs0 = BL.q(x0)
################## prior specification with θ = (a, c, γ) ########################
#priorθ = product_distribution(fill(Exponential(1.0),3))
priorθ = product_distribution([Exponential(ainit), Exponential(cinit), Exponential(γinit)])
κ = 100.0
prior_momenta = MvNormalCanon(gramkernel(xobs0,P)/κ)
logpriormom(x0) = logpdf(prior_momenta, vcat(BL.p(x0)...))# +logpdf(prior_positions, vcat(BL.q(x0)...))

#########################

xinit = State(x0.q, zeros(PointF,n))#x0
mT = zeros(PointF,n)


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
write_acc(accinfo,accpcn,nshapes,outdir)

v0 = Vector(xobs0)
vT = [Vector(xobsT[1])]
write_observations(v0, vT, n, nshapes, x0,outdir)
write_noisefields(P,outdir)
