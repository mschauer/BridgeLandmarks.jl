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
[:mala_mom, :rmmala_mom, :rmrw_mom, :mala_pos, :rmmala_pos, :parameter, :innov]

dat = load("../experiments/exp1/data_exp1.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]

@time landmarksmatching(xobs0,xobsT;outdir=outdir,ITER=25,
            updatescheme = [:innov, :rmmala_mom, :parameter],
                pars= Pars_ms(covθprop = Diagonal(fill(0.0001,3))))
#@enter landmarksmatching(xobs0,xobsT)





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


################ another example ################################################################
n = 7

a = 2.0     # Hamiltonian kernel parameter (the larger, the stronger landmarks behave similarly)
c = 0.1     # multiplicative factor in kernel
γ = 0.7     # Noise level

Ptrue = MarslandShardlow(a, c, γ, 0.0, n)

xobs0 = [PointF(2.0cos(t), sin(t))/4.0  for t in collect(0:(2pi/n):2pi)[2:end]]

θ, ψ =  π/6, 0.1
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
shift = [0.5, 1.0]
xobsT = [rot * stretch * xobs0[i] + shift  for i in 1:n ]
ainit = mean(norm.([xobs0[i]-xobs0[i-1] for i in 2:n]))/5.0

@time landmarksmatching(xobs0,xobsT;outdir=outdir,ITER=10,updatescheme = [:innov,:rmmala_mom],#, :parameter],
                pars= Pars_ms(δmom=0.001, ρinit=0.98,σobs=0.001,γinit=0.01),ainit=ainit)





dat = load("../experiments/exp2/data_exp2.jld2")
xobsT = dat["xobsT"]
#xobs0 = dat["xobs0"]

template_estimation(xobsT;ITER=150)
