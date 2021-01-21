using Revise
using BridgeLandmarks
const BL = BridgeLandmarks
using Distributions
using LinearAlgebra
using Random
using DelimitedFiles
using ReverseDiff
using ForwardDiff
using Zygote
using StaticArrays

n = 101
xobs0 = [PointF(2.0cos(t), sin(t))/4.0  for t in collect(0:(2pi/n):2pi)[2:end]]
θ, ψ =  π/6, 0.1
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
shift = [0.5, -1.0]
xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:n]

adaptskip = 1
skip_saveITER = 50
printskip = 5#1000
ITER = 100#25_000

δmom = [0.01]

p_ahs = Pars_ahs(δmom=δmom,  db=[2.0, 1.0],stdev=.2,γinit=0.1, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01, σobs=0.01,
                                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.995)

p_ms = Pars_ms(δmom=δmom,  γinit=1.0/√n, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01, σobs=0.01,
                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.999)
pars = p_ms
model = pars.model
n = length(xobs0)
nshapes = 1

Σobs = fill([pars.σobs^2 * one(UncF) for i in 1:n],2) # noise on observations
T = 1.0; t = 0.0:pars.dt:T; tt = tc(t,T)
obs_atzero = true
fixinitmomentato0 = false
subsamples = 0:pars.skip_saveITER:ITER
obsinfo = BL.set_obsinfo(xobs0,[xobsT],Σobs, obs_atzero,fixinitmomentato0)

################################# initialise P #################################
ainit = 0.5*mean(norm.(diff(xobs0)))
cinit = pars.cinit
γinit = pars.γinit
if model == :ms
    P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
elseif model == :ahs
    nfsinit = construct_nfs(pars.db, pars.stdev, γinit)
    P = Landmarks(ainit, cinit, n, pars.db , pars.stdev, nfsinit)
end

################## prior specification with θ = (a, c, γ) ########################
priorθ = product_distribution([pars.aprior, pars.γprior])
priormom = MvNormalCanon(gramkernel(xobs0,P)/pars.κ)

xinit = State(xobs0, zeros(PointF,P.n))
mT = zeros(PointF, n)


### from lmguid.jl ###
lt = length(t)
StateW = PointF
dwiener = BL.dimwiener(P)
nshapes = obsinfo.nshapes

guidrec = [BL.GuidRecursions(t,obsinfo)  for _ ∈ 1:nshapes]  # initialise guiding terms
gramT_container = [BL.gram_matrix(obsinfo.xobsT[k], P) for k ∈ 1:nshapes]  # gram matrices at observations at time T

Paux = [BL.auxiliary(P, State(obsinfo.xobsT[k],mT), gramT_container[k]) for k ∈ 1:nshapes] # auxiliary process for each shape
Q = BL.GuidedProposal(P,Paux,t,obsinfo.xobs0,obsinfo.xobsT,guidrec,nshapes,[mT for _ ∈ 1:nshapes])
Q = BL.update_guidrec!(Q, obsinfo)   # compute backwards recursion

X = [BL.initSamplePath(t, xinit) for _ ∈ 1:nshapes]
W = [BL.initSamplePath(t,  zeros(StateW, dwiener)) for _ ∈ 1:nshapes]
for k ∈ 1:nshapes   sample!(W[k], BL.Wiener{Vector{StateW}}())  end

x = deepvec(xinit)
q, p = BL.split_state(xinit)
qᵒ = deepcopy(q); pᵒ = deepcopy(q); ∇ = deepcopy(q); ∇ᵒ = deepcopy(q)

# memory allocations, actual state at each iteration is (X, W, Q, x, ∇x) (x, ∇x are initial state and its gradient)
Xᵒ = deepcopy(X)
Wᵒ = BL.initSamplePath(t,  zeros(StateW, dwiener))
Wnew = BL.initSamplePath(t,  zeros(StateW, dwiener))
# sample guided proposal and compute loglikelihood (write into X)
At = zeros(UncF, 2P.n, 2P.n)  # container for to compute a=σσ' for auxiliary process.
@time ll, X = BL.gp!(BL.LeftRule(), X, xinit, W, Q, At; skip=sk)

# setup containers for saving objects
Xsave = typeof(zeros(length(t) * P.n * 2 * d * nshapes))[]
push!(Xsave, BL.convert_samplepath(X))
parsave = Vector{Float64}[]
push!(parsave, BL.getpars(Q))
initendstates_save = [BL.extract_initial_and_endstate(0,X[1]) ]

δ = [pars.δpos, pars.δmom, pars.δsgd_mom]
δa, δγ = pars.δa, pars.δγ
ρ = pars.ρlowerbound
adapt(i) = (i > 1.5*pars.adaptskip) & (mod(i,pars.adaptskip)==0)

x0 = X[1].yy[1]
dK = BL.gramkernel(x0.q, P)
inv_dK = inv(dK)

P = Q.target;   n = P.n
x0 = BL.deepvec2state(x)
q, p = BL.split_state(x0)
dn = d * n

u = BL.slogρ_mom!(q, Q, W, X, priormom,ll, At)
cfg = ForwardDiff.GradientConfig(u, p, ForwardDiff.Chunk{dn}()) # d*P.n is maximal

#using Profile
#Profile.clear()
#@profile ForwardDiff.gradient!(∇, u, p, cfg)
cfg = ForwardDiff.GradientConfig(u, p, ForwardDiff.Chunk{dn}())
@time ForwardDiff.gradient!(∇, u, p, cfg)
#Juno.profiler()








# ftape = ReverseDiff.GradientTape(u, p)
# compiled_ftape = ReverseDiff.compile(ftape)
# @time ReverseDiff.gradient!(∇, compiled_ftape, p)
