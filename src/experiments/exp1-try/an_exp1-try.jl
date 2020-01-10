# ellipse to rotated and shifted ellipse
# initial and final landmark positions observed
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

Random.seed!(9)

workdir = @__DIR__
cd(workdir)

pyplot()
include(dirname(dirname(workdir))*"/plotting.jl")
include(dirname(dirname(workdir))*"/postprocessing.jl")
outdir = workdir*("/")

#Random.seed!(3)

#-------- read data ----------------------------------------------------------
dat = load("data_exp1-try.jld")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
x0 = dat["x0"]
nshapes = dat["nshapes"]

xobsT = BL.circshift(xobsT,true)
xobsT = BL.circshift(xobsT,true)
################################# start settings #################################
models = [:ms, :ahs]
model = models[1]
sampler = :mcmc

fixinitmomentato0 = false
obs_atzero = true
if model==:ms
    #σobs = 0.01   # noise on observations
    σobs = 0.01
    Σobs = [σobs^2 * one(UncF) for i in 1:n]
    # σobsv = vcat(fill(σobs,9), fill(0.1,n-9))
    # Σobs = [σobsv[i]^2 * one(UncF) for i in 1:n]
else
    σobs = 0.01   # noise on observations
    Σobs = [σobs^2 * one(UncF) for i in 1:n]
end
T = 1.0
dt = 0.01
t = 0.0:dt:T; tt_ =  tc(t,T)
updatepars =  false#true#false

make_animation = false

ITER = 50
subsamples = 0:1:ITER
adaptskip = 20  # adapt mcmc tuning pars every adaptskip iters
maxnrpaths = 10 # update at most maxnrpaths Wiener increments at once

#-------- set prior on θ = (a, c, γ) ----------------------------------------------------------
prior_a = Exponential(1.0)
prior_c = Exponential(1.0)
prior_γ = Exponential(1.0)

#--------- MCMC tuning pars ---------------------------------------------------------
initstate_updatetypes =  [:mala_mom, :rmmala_pos] #, [:rmmala_mom]

ρinit = 0.9              # pcN-step
σ_a = 0.2  # update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())
σ_c = 0.2  # update c to cᵒ as cᵒ = c * exp(σ_c * rnorm())
σ_γ = 0.2  # update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())
if model==:ms
    δinit = [0.01, 0.1] # first comp is not used
else
    δinit = [0.1, 0.1] # first comp is not used
end
η(n) = min(0.2, 10/n)  # adaptation rate for adjusting tuning pars
################################# end settings #################################

ainit = mean(norm.([x0.q[i]-x0.q[i-1] for i in 2:n]))/2.0   # Let op: door 2 gedeeld



if model == :ms
    cinit = 0.2
    γinit = 2.0
    P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
elseif model == :ahs
    cinit = 0.02#0.05
    γinit = 0.2
    stdev = 0.75
    nfsinit = construct_nfs(2.5, stdev, γinit)
    P = Landmarks(ainit, cinit, n, 2.5, stdev, nfsinit)
end

#--------- set prior on momenta -------------------------
κ = 100.0
prior_momenta = MvNormalCanon(gramkernel(xobs0,P)/κ)
prior_positions = MvNormal(vcat(xobs0...), σobs)
logpriormom(x0) = logpdf(prior_momenta, vcat(BL.p(x0)...))# +logpdf(prior_positions, vcat(BL.q(x0)...))


mT = zeros(PointF,n)   # vector of momenta at time T used for constructing guiding term #mT = randn(PointF,P.n)

start = time() # to compute elapsed time
    xobsT = [xobsT]
    xinit = State(xobs0, zeros(PointF,P.n))
    #xinit = State(xobs0, rand(PointF,P.n))

    anim, Xsave, parsave, objvals, accpcn, accinfo, δ, ρ = lm_mcmc(tt_, (xobs0,xobsT), Σobs, mT, P,
             sampler, obs_atzero, fixinitmomentato0,
             xinit, ITER, subsamples,
            (ρinit, maxnrpaths, δinit, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ, η),
            logpriormom,
            initstate_updatetypes, adaptskip,
            outdir,  dat["pb"]; updatepars = updatepars, make_animation=make_animation)
elapsed = time() - start

#----------- post processing -------------------------------------------------
println("Elapsed time: ",round(elapsed/60;digits=2), " minutes")
perc_acc_pcn = mean(accpcn)*100
println("Acceptance percentage pCN step: ", round(perc_acc_pcn;digits=2))

write_mcmc_iterates(Xsave, tt_, n, nshapes, subsamples, outdir)
write_info(sampler, ITER, n, tt_, Σobs, ρinit, δinit,ρ, δ, perc_acc_pcn,
updatepars, model, adaptskip, maxnrpaths, initstate_updatetypes, outdir)
write_observations(xobs0, xobsT, n, nshapes, x0,outdir)
write_acc(accinfo,accpcn,nshapes,outdir)
write_params(parsave,subsamples,outdir)
write_noisefields(P,outdir)
if make_animation
    fn = string(model)
    gif(anim, outdir*"anim.gif", fps = 50)
    mp4(anim, outdir*"anim.mp4", fps = 50)
end
