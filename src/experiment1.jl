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
n = 10  # nr of landmarks
models = [:ms, :ahs]
model = models[1]
sampler = :mcmc
fixinitmomentato0 = false
obs_atzero = true
σobs = 0.01   # noise on observations
T = 1.0
dt = 0.01
t = 0.0:dt:T; tt_ =  tc(t,T)
updatepars = true

ITER = 250
subsamples = 0:1:ITER
adaptskip = 10

#-------- set prior on θ = (a, c, γ) ----------------------------------------------------------
prior_a = Exponential(5.0)
prior_c = Exponential(5.0)
prior_γ = Exponential(5.0)

#-------- generate data ----------------------------------------------------------
dataset = "shiftedextreme"
a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
c = 0.1     # multiplicative factor in kernel
γ = 1.0     # Noise level
if model == :ms
    λ = 0.0;
    nfs = 0
    Ptrue = MarslandShardlow(a, c, γ, λ, n)
else
    db = 2.0 # domainbound
    nfstd = 1.0
    nfs = construct_nfs(db, nfstd, γ)
    Ptrue = Landmarks(a, c, n, db, nfstd, nfs)
end
q0 = [PointF(2.0cos(t), sin(t))  for t in (0:(2pi/n):2pi)[1:n]]
p0 = [PointF(1.0, -3.0) for i in 1:n]/n
x0 = State(q0, p0)
Wf, Xf = landmarksforward(0.0:0.001:T, x0, Ptrue)
xobs0 = x0.q + σobs * randn(PointF,n)
θ, η =  π/5, 0.4
pb = Lmplotbounds(-3.0,3.0,-3.0,3.0)
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + η, 0.0, 0.0, 1.0 - η)
xobsT = [rot * stretch * xobs0[i]  for i in 1:Ptrue.n ] + σobs * randn(PointF,n)

#--------- MCMC tuning pars ---------------------------------------------------------
ρ = 0.8              # pcN-step
σ_a = 0.2  # update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())
σ_c = 0.2  # update c to cᵒ as cᵒ = c * exp(σ_c * rnorm())
σ_γ = 0.2  # update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())

################################# end settings #################################

#---------------- step-size on mala steps for initial state -------------------
δ = [0.0, 0.01] # first comp is not used


#(ainit, cinit, γinit) = (1.0, 1.0, 1.0)
ainit = mean(norm.([q0[i]-q0[i-1] for i in 2:n]))
cinit = 1.0
γinit = 0.1
if model == :ms
    P = MarslandShardlow(ainit, cinit, γinit, Ptrue.λ, Ptrue.n)
elseif model == :ahs
    nfsinit = construct_nfs(Ptrue.db, Ptrue.nfstd, γinit)
    P = Landmarks(ainit, cinit, Ptrue.n, Ptrue.db, Ptrue.nfstd, nfsinit)
end

mT = zeros(PointF,Ptrue.n)   # vector of momenta at time T used for constructing guiding term #mT = randn(PointF,P.n)

start = time() # to compute elapsed time
    xobsT = [xobsT]
    xinit = State(xobs0, zeros(PointF,P.n))
    initstate_updatetypes = [:mala_mom]
    anim, Xsave, parsave, objvals, perc_acc_pcn, accinfo = lm_mcmc(tt_, (xobs0,xobsT), σobs, mT, P,
             sampler, obs_atzero, fixinitmomentato0,
             xinit, ITER, subsamples,
            (ρ, δ, prior_a, prior_c, prior_γ, σ_a, σ_c, σ_γ), initstate_updatetypes, adaptskip,
            outdir,  pb; updatepars = updatepars, makefig=true, showmomenta=false)
elapsed = time() - start

#----------- post processing -------------------------------------------------
println("Elapsed    time: ",round(elapsed/60;digits=2), " minutes")
println("Acceptance percentage pCN step: ",perc_acc_pcn)
include("./postprocessing.jl")
