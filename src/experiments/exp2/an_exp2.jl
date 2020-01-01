# ellipse to rotated and shifted ellipse
# initial and final landmark positions observed

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
cd(workdir)

pyplot()

include(dirname(dirname(workdir))*"/plotting.jl")
include(dirname(dirname(workdir))*"/postprocessing.jl")
outdir = workdir*("/")

Random.seed!(3)

################################# start settings #################################
models = [:ms, :ahs]
model = models[1]
sampler = :mcmc

fixinitmomentato0 = true
if fixinitmomentato0
    initstate_updatetypes = [:rmmala_pos]
else
    initstate_updatetypes = [:mala_mom, :rmmala_pos]
end

obs_atzero = false
σobs = 0.01   # noise on observations
T = 1.0
dt = 0.01
t = 0.0:dt:T; tt_ =  tc(t,T)
updatepars = true

make_animation = false

ITER = 750
subsamples = 0:1:ITER
adaptskip = 20  # adapt mcmc tuning pars every adaptskip iters
maxnrpaths = 10 # update at most maxnrpaths Wiener increments at once

#-------- set prior on θ = (a, c, γ) ----------------------------------------------------------
prior_a = Exponential(1.0)
prior_c = Exponential(1.0)
prior_γ = Exponential(1.0)

#-------- generate data ----------------------------------------------------------
dat = load("data_exp2.jld")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
x0 = dat["x0"]
nshapes = dat["nshapes"]

#--------- MCMC tuning pars ---------------------------------------------------------

ρinit = 0.7              # pcN-step
σ_a = 0.2  # update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())
σ_c = 0.2  # update c to cᵒ as cᵒ = c * exp(σ_c * rnorm())
σ_γ = 0.2  # update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())
if model==:ms
    δinit = [0.002, 0.1]
else
    δinit = [0.01, 0.1]
end
η(n) = min(0.2, 10/n)  # adaptation rate for adjusting tuning pars
################################# end settings #################################

ainit = mean(norm.([x0.q[i]-x0.q[i-1] for i in 2:n]))/2.0
if model == :ms
    cinit = 0.2
    γinit = 2.0
    P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
elseif model == :ahs
    cinit = 0.05
    γinit = 0.5
    stdev = 0.75
    nfsinit = construct_nfs(2.5, stdev, γinit)
    P = Landmarks(ainit, cinit, n, 2.5, stdev, nfsinit)
end

mT = zeros(PointF,n)   # vector of momenta at time T used for constructing guiding term #mT = randn(PointF,P.n)
# deliberately take wrong initial landmark configuration
xinitq = xobsT[1]
θ, ψ =  π/6, 0.25
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
xinitq_adj = [rot * stretch * xinitq[i] for i in 1:n ]
xinit = State(xinitq_adj, mT)


start = time() # to compute elapsed time

    anim, Xsave, parsave, objvals, accpcn, accinfo, δ, ρ = lm_mcmc(tt_, (xobs0,xobsT), σobs, mT, P,
             sampler, obs_atzero, fixinitmomentato0,
             xinit, ITER, subsamples,
            (ρinit, maxnrpaths, δinit, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ, η), initstate_updatetypes, adaptskip,
            outdir,  dat["pb"]; updatepars = updatepars, make_animation=make_animation)
elapsed = time() - start

#----------- post processing -------------------------------------------------
println("Elapsed time: ",round(elapsed/60;digits=2), " minutes")
perc_acc_pcn = mean(accpcn)*100
println("Acceptance percentage pCN step: ", round(perc_acc_pcn;digits=2))


write_mcmc_iterates(Xsave, tt_, n, nshapes, subsamples, outdir)
write_info(sampler, ITER, n, tt_,σobs, ρinit, δinit,ρ, δ, perc_acc_pcn,updatepars, model, adaptskip, maxnrpaths, initstate_updatetypes, outdir)
write_observations(xobs0, xobsT, n, nshapes, x0,outdir)
write_acc(accinfo,accpcn,nshapes,outdir)
write_params(parsave,subsamples,outdir)
write_noisefields(P,outdir)
if make_animation
    fn = string(model)
    gif(anim, outdir*"anim.gif", fps = 50)
    mp4(anim, outdir*"anim.mp4", fps = 50)
end
