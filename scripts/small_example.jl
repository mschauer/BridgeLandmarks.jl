using Revise
using BridgeLandmarks
const BL = BridgeLandmarks
using Distributions
using LinearAlgebra
using Random
using StaticArrays

# A small example useful for testing purposes

n = 27
xobs0 = [PointF(2.0cos(t), sin(t))/4.0  for t in collect(0:(2pi/n):2pi)[2:end]]
θ, ψ =  π/6, 0.1
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
shift = [0.5, -1.0]
xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:n]


pars = Pars_ms(γinit=1.0/√n, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01, ρlowerbound=0.9)
pars = Pars_ahs(db=[3.0, 2.0],stdev=.5,γinit=.1, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01, ρlowerbound=0.9)

Σobs = fill([pars.σobs^2 * one(UncF) for i in 1:n],2) # noise on observations

T = 1.0; t = 0.0:pars.dt:T; tt = tc(t,T)
obs_atzero = true
fixinitmomentato0 = false
ITER = 100
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
gramT_container = [BL.gram_matrix([xobsT][k], P) for k ∈ 1:nshapes]  # gram matrices at observations at time T

Paux = [BL.auxiliary(P, State(obsinfo.xobsT[k],mT), gramT_container[k]) for k ∈ 1:nshapes]
guidrec = [BL.GuidRecursions(t,obsinfo)  for _ ∈ 1:nshapes]  # initialise guiding terms
Paux = [BL.auxiliary(P, State(obsinfo.xobsT[k],mT)) for k ∈ 1:nshapes] # auxiliary process for each shape
Q = BL.GuidedProposal(P,Paux,t,obsinfo.xobs0,obsinfo.xobsT,guidrec,nshapes,[mT for _ ∈ 1:nshapes])
Q = BL.update_guidrec!(Q, obsinfo)   # compute backwards recursion

X = [BL.initSamplePath(t, xinit) for _ ∈ 1:nshapes]
W = [BL.initSamplePath(t,  zeros(PointF, BL.dimwiener(P))) for _ ∈ 1:nshapes]
for k ∈ 1:nshapes   sample!(W[k], BL.Wiener{Vector{PointF}}())  end

ll, X = BL.gp!(BL.LeftRule(), X, xinit, W, Q; skip=sk)
#only one shape
Xpath = X[1]


a = [extract_initial_and_endstate(5, Xpath)]
push!(a, extract_initial_and_endstate(555, Xpath))
A =vcat(a...)
B = DataFrame(A)
rename!(B, ["iterate", "time0", "time1"])
N = div(nrow(B),2d)
B[!,:type] = repeat(["pos1", "pos2", "mom1", "mom2"], N)

function testf!(X, xinit, W, Q, R; skip=sk)
    for i in 1:R
        ll, X = BL.gp!(BL.LeftRule(), X, xinit, W, Q; skip=sk)
    end
    ll, X
end
@btime testf!(X, xinit, W, Q, 1; skip=sk)
@time testf!(X, xinit, W, Q, 10; skip=sk)


function testf2!(X, xinit, W, Q, R; skip=sk)
    for i in 1:R
        BL.gp!(BL.LeftRule(), X, xinit, W, Q; skip=sk)
    end
    nothing
end
@btime testf2!(X, xinit, W, Q, 1; skip=sk)




# check vecofpoints2state


xv = rand(PointF, 10)

function opt1(xv)
    BL.deepvec2state(BL.deepvec(xv))
end

@btime  opt1(xv)
@btime  BL.vecofpoints2state(xv)



nshapes = 1
T = 1.0; t = 0.0:pars.dt:T; tt = tc(t,T)
#W = [BL.initSamplePath(t,  zeros(PointF, BL.dimwiener(P))) for _ ∈ 1:nshapes]

W = BL.initSamplePath(t,  zeros(PointF, BL.dimwiener(P)))
Wnew = BL.initSamplePath(t,  zeros(PointF, BL.dimwiener(P)))
Wᵒ = BL.initSamplePath(t,  zeros(PointF, BL.dimwiener(P)))


function update_path2!(W, Wnew,Wᵒ)
    ρ = 0.9
    ρ_ = sqrt(1.0-ρ^2)
    # From current state (x,W) with loglikelihood ll, update to (x, Wᵒ)
#    sample!(Wnew, Bridge.Wiener{Vector{PointF}}())
    for i ∈ eachindex(W.yy)
        for j ∈ eachindex(W.yy[1])
            Wᵒ.yy[i][j] = ρ * W.yy[i][j] + ρ_ * Wnew.yy[i][j]
        end
    end
end


function ff2(W, Wnew, Wᵒ, R)
    for k in 1:R
        update_path2!(W, Wnew, Wᵒ)
    end
end
@btime ff2(W, Wnew, Wᵒ, 1)
@btime ff2(W, Wnew, Wᵒ, 2)
@btime ff2(W, Wnew, Wᵒ, 3)


function update_path3!(W, Wnew,Wᵒ)
    ρ = 0.9
    ρ_ = sqrt(1.0-ρ^2)
    # From current state (x,W) with loglikelihood ll, update to (x, Wᵒ)
#    sample!(Wnew, Bridge.Wiener{Vector{PointF}}())
    Wᵒ.yy .= ρ * W[1].yy .+ ρ_ * Wnew.yy
end


function ff3(W, Wnew, Wᵒ, R)
    for k in 1:R
        update_path2!(W, Wnew, Wᵒ)
    end
end
@btime ff3(W, Wnew, Wᵒ, 1)
@btime ff3(W, Wnew, Wᵒ, 2)


@btime landmarksmatching(xobs0,xobsT; ITER=10, pars=p_ms, updatescheme=ups, printskip=printskip, outdir=outdir_ms)
#  259.032 ms (2205591 allocations: 231.72 MiB)
# 273.766 ms (2205601 allocations: 231.72 MiB)
#  1.023 s (21896571 allocations: 1.04 GiB)  (also calling gp!)
#  201.703 ms (236768 allocations: 151.89 MiB)
@btime landmarksmatching(xobs0,xobsT; ITER=20, pars=p_ms, updatescheme=ups, printskip=printskip, outdir=outdir_ms)
#  268.554 ms (2229751 allocations: 232.47 MiB)
#   264.369 ms (2229771 allocations: 232.47 MiB)
#  1.863 s (41611713 allocations: 1.84 GiB)  (also calling gp!)
# 220.015 ms (261200 allocations: 155.45 MiB)


struct MatArr{T}
    x::Vector{Matrix{T}}
end

u1 = MatArr(fill(zeros(Float64,2,3),2))
u1.x[1] = reshape(rand(4),2,2)
