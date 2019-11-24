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
println(workdir)
cd(workdir)

pyplot()
include("plotting.jl")
include("postprocessing.jl")
outdir = "./figs/"

Random.seed!(3)

################################# start settings #################################
models = [:ms, :ahs]
model = models[1]
sampler = :mcmc
nshapes = 1
fixinitmomentato0 = false
obs_atzero = false
σobs = 0.01   # noise on observations
T = 1.0
dt = 0.01
t = 0.0:dt:T; tt_ =  tc(t,T)
updatepars = true

make_animation = false

ITER = 250
subsamples = 0:1:ITER
adaptskip = 10  # adapt mcmc tuning pars every adaptskip iters

#-------- set prior on θ = (a, c, γ) ----------------------------------------------------------
prior_a = Exponential(1.0)
prior_c = Exponential(1.0)
prior_γ = Exponential(1.0)

#-------- generate data ----------------------------------------------------------
dat = load("./figs/data_exp2.jld")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
x0 = dat["x0"]
nshapes = dat["nshapes"]

#--------- MCMC tuning pars ---------------------------------------------------------
ρ = 0.8              # pcN-step
σ_a = 0.2  # update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())
σ_c = 0.2  # update c to cᵒ as cᵒ = c * exp(σ_c * rnorm())
σ_γ = 0.2  # update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())
δ = [0.0005, 0.005]

η(n) = min(0.1, 10/sqrt(n))  # adaptation rate for adjusting tuning pars
################################# end settings #################################

ainit = mean(norm.([x0.q[i]-x0.q[i-1] for i in 2:n]))
cinit = 0.2
γinit = 0.1
if model == :ms
    P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
elseif model == :ahs
    stdev = 0.75
    nfsinit = construct_nfs(2.5, stdev, γinit)
    P = Landmarks(ainit, cinit, n, 2.5, stdev, nfsinit)
end

mT = zeros(PointF,n)   # vector of momenta at time T used for constructing guiding term #mT = randn(PointF,P.n)

start = time() # to compute elapsed time
    xinitq = xobsT[1]
    xinit = State(xinitq, zeros(PointF,P.n))
    initstate_updatetypes = [:mala_mom, :rmmala_pos]
    anim, Xsave, parsave, objvals, acc_pcn, accinfo = lm_mcmc(tt_, (xobs0,xobsT), σobs, mT, P,
             sampler, obs_atzero, fixinitmomentato0,
             xinit, ITER, subsamples,
            (ρ, δ, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ, η), initstate_updatetypes, adaptskip,
            outdir,  dat["pb"]; updatepars = updatepars, make_animation=make_animation)
elapsed = time() - start

#----------- post processing -------------------------------------------------
println("Elapsed time: ",round(elapsed/60;digits=2), " minutes")
perc_acc_pcn = mean(acc_pcn)*100
println("Acceptance percentage pCN step: ", round(perc_acc_pcn;digits=2))

write_mcmc_iterates(Xsave, tt_, n, nshapes, subsamples, outdir)
write_info(sampler, ITER, n, tt_,σobs, ρ, δ, perc_acc_pcn, outdir)
write_observations(xobs0, xobsT, n, nshapes, x0,outdir)
write_acc_par_and_initstate(accinfo,outdir)
write_params(parsave,subsamples,outdir)
write_noisefields(P,outdir)
if make_animation
    fn = string(model)
    gif(anim, outdir*"anim.gif", fps = 50)
    mp4(anim, outdir*"anim.mp4", fps = 50)
end
