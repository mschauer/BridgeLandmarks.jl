using BridgeLandmarks
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

workdir = @__DIR__
println(workdir)
cd(workdir)

pyplot()
include("plotting.jl")
outdir = "./figs/"

Random.seed!(3)

################################# start settings #################################
models = [:ms, :ahs]
model = models[2]
sampler = :mcmc
fixinitmomentato0 = false
obs_atzero = true
σobs = 0.01   # noise on observations
T = 1.0
dt = 0.01
t = 0.0:dt:T; tt_ =  tc(t,T)
updatepars = false#true

ITER = 100
subsamples = 0:1:ITER
adaptskip = 10  # adapt mcmc tuning pars every adaptskip iters

#-------- set prior on θ = (a, c, γ) ----------------------------------------------------------
prior_a = Exponential(1.0)
prior_c = Exponential(1.0)
prior_γ = Exponential(1.0)

#-------- generate data ----------------------------------------------------------
dat = load("./figs/data_exp1.jld")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
x0 = dat["x0"]

#--------- MCMC tuning pars ---------------------------------------------------------
ρ = 0.8              # pcN-step
σ_a = 0.2  # update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())
σ_c = 0.2  # update c to cᵒ as cᵒ = c * exp(σ_c * rnorm())
σ_γ = 0.2  # update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())
δ = [0.0, 0.01] # first comp is not used
################################# end settings #################################

ainit = mean(norm.([x0.q[i]-x0.q[i-1] for i in 2:n]))
cinit = 0.2
γinit = 0.1
if model == :ms
    P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
elseif model == :ahs
    nfsinit = construct_nfs(2.5, 0.6, γinit)
    P = Landmarks(ainit, cinit, n, 2.5, 0.6, nfsinit)
end

mT = zeros(PointF,n)   # vector of momenta at time T used for constructing guiding term #mT = randn(PointF,P.n)

start = time() # to compute elapsed time
    xobsT = [xobsT]
    xinit = State(xobs0, zeros(PointF,P.n))
    initstate_updatetypes = [:mala_mom]
    anim, Xsave, parsave, objvals, acc_pcn, accinfo = lm_mcmc(tt_, (xobs0,xobsT), σobs, mT, P,
             sampler, obs_atzero, fixinitmomentato0,
             xinit, ITER, subsamples,
            (ρ, δ, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ), initstate_updatetypes, adaptskip,
            outdir,  dat["pb"]; updatepars = updatepars, makefig=true, showmomenta=false)
elapsed = time() - start

#----------- post processing -------------------------------------------------
println("Elapsed time: ",round(elapsed/60;digits=2), " minutes")
perc_acc_pcn = mean(acc_pcn)*100
println("Acceptance percentage pCN step: ", round(perc_acc_pcn;digits=2))

include("./postprocessing.jl")
