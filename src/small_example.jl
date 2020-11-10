# A small example useful for testing purposes

n = 7
xobs0 = [PointF(2.0cos(t), sin(t))/4.0  for t in collect(0:(2pi/n):2pi)[2:end]]
θ, ψ =  π/6, 0.1
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
shift = [0.5, -1.0]
xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:n]


pars = Pars_ms(γinit=1.0/√n, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01, ρlowerbound=0.9)
pars = Pars_ahs(db=[3.0, 2.0],stdev=.5,γinit=.1, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01, ρlowerbound=0.9)

Σobs = fill([pars.σobs^2 * one(UncF) for i in 1:n],2) # noise on observations
end
T = 1.0; t = 0.0:pars.dt:T; tt = tc(t,T)
obs_atzero = true
fixinitmomentato0 = false
subsamples = 0:pars.skip_saveITER:ITER
obsinfo = BL.set_obsinfo(xobs0,[xobsT],Σobs, obs_atzero,fixinitmomentato0)

ainit = 0.5*mean(norm.(diff(xobs0)))
cinit = pars.cinit
γinit = pars.γinit
if pars.model == :ms
    P = MarslandShardlow(ainit, cinit, γinit, 0.0, n)
elseif pars.model == :ahs
    nfsinit = construct_nfs(pars.db, pars.stdev, γinit)
    P = Landmarks(ainit, cinit, n, pars.db , pars.stdev, nfsinit)
end

xinit = State(xobs0, zeros(PointF,P.n))
mT = zeros(PointF, n)

nshapes = 1
guidrec = [BL.GuidRecursions(t,obsinfo)  for _ ∈ 1:nshapes]  # initialise guiding terms
Paux = [BL.auxiliary(P, State(obsinfo.xobsT[k],mT)) for k ∈ 1:nshapes] # auxiliary process for each shape
Q = BL.GuidedProposal(P,Paux,t,obsinfo.xobs0,obsinfo.xobsT,guidrec,nshapes,[mT for _ ∈ 1:nshapes])
Q = BL.update_guidrec!(Q, obsinfo)   # compute backwards recursion

X = [BL.initSamplePath(t, xinit) for _ ∈ 1:nshapes]
W = [BL.initSamplePath(t,  zeros(PointF, BL.dimwiener(P))) for _ ∈ 1:nshapes]
for k ∈ 1:nshapes   sample!(W[k], BL.Wiener{Vector{PointF}}())  end

ll, X = BL.gp!(BL.LeftRule(), X, xinit, W, Q; skip=sk)
