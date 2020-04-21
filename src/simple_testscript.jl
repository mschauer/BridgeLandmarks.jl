using Revise

using BridgeLandmarks
const BL = BridgeLandmarks
using RCall
using Random
using Distributions
using DataFrames
using DelimitedFiles
using CSV
using StaticArrays
using LinearAlgebra
using JLD2
using FileIO
using Parameters
#using Debugger

Random.seed!(9)
workdir = @__DIR__
cd(workdir)
outdir = joinpath(workdir,"out")

## available update steps implemented:
println("available steps are [:mala_mom, :rmmala_mom, :rmrw_mom, :mala_pos, :rmmala_pos, :sgd_mom :parameter, :innov]")

####################### test landmarksmatching ######################
n = 8
xobs0 = [PointF(2.0cos(t), sin(t))/4.0  for t in collect(0:(2pi/n):2pi)[2:end]]
θ, ψ =  π/6, 0.1
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
shift = [0.5, -1.0]
xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:n]

@time landmarksmatching(xobs0,xobsT; ITER=100,pars=Pars_ms(δmom=0.01), outdir=outdir)

@time landmarksmatching(xobs0,xobsT; ITER=300,pars=Pars_ms(δmom=0.001), outdir=outdir,updatescheme=[:innov, :sgd_mom, :parameter])

@time landmarksmatching(xobs0,xobsT; updatescheme=[:innov, :rmrw_mom], ITER=500,pars=Pars_ms(δmom=0.01), outdir=outdir)

landmarksmatching(xobs0,xobsT; outdir=outdir,ITER=500,
                            updatescheme = [:innov, :parameter,:mala_mom, :matching],
                            pars= Pars_ms(covθprop = Diagonal(fill(0.001,3)),η = n -> min(0.2, 10/(n))  ))

plotlandmarksmatching(outdir)
###################### test template estimation ######################
n = 6
nshapes = 8
q0 = [PointF(2.0cos(t), sin(t))  for t in collect(0:(2pi/n):2pi)[2:end]] # initial shape is an ellipse
x0 = State(q0, randn(PointF,n))
xobsT = Vector{PointF}[]
σobs = 0.05

T = 1.0; dt = 0.01; t = 0.0:dt:T
Ptrue = MarslandShardlow(2.0, 0.1, 1.7, 0.0, n)
for k in 1:nshapes
    Wf, Xf = BridgeLandmarks.landmarksforward(t, x0, Ptrue)
    push!(xobsT, Xf.yy[end].q + σobs * randn(PointF,n))
end


template_estimation(xobsT; xinitq=2.0*xobsT[1],outdir=outdir)  # deliberately initialise badly to show it works

# cyclically shift landmarks for a few landmarks
xobsTshift = copy(xobsT)
xobsTshift[1] = Base.circshift(xobsT[1],2)
xobsTshift[2] = Base.circshift(xobsT[2],-1)
xobsTshift[3] = Base.circshift(xobsT[3],-3)
template_estimation(xobsTshift; xinitq=2.0*xobsT[3],outdir=outdir, updatescheme = [:innov, :rmmala_pos, :parameter, :matching],
    ITER=200, pars=Pars_ms(δpos=0.0001,skip_saveITER=5))  # deliberately initialise badly to show it works

####################### below code for generating input data as matrices for landmarksmatching ######################
if false
    dat = load("../experiments/exp1/data_exp1.jld2")
    xobs0 = dat["xobs0"]
    xobsT = dat["xobsT"]
    if false # make example dataset
        writedlm("landmarks0.txt", hcat(extractcomp(xobs0,1), extractcomp(xobs0,2)))
        writedlm("landmarksT.txt", hcat(extractcomp(xobsT,1), extractcomp(xobsT,2)))
    end
    landmarks0 = readdlm("landmarks0.txt")
    landmarksT = readdlm("landmarksT.txt")

    landmarksmatching(xobs0,xobsT)
    landmarksmatching(landmarks0, landmarksT; outdir=outdir, ITER=10)
    landmarksmatching(landmarks0, landmarksT; outdir=outdir, ITER=10, pars=BL.Pars_ahs())
end
