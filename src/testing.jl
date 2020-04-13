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


dat = load("../experiments/exp1/data_exp1.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
@time landmarksmatching(xobs0,xobsT;outdir=outdir,ITER=100,updatescheme = [:innov, :rmmala_mom, :parameter],
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

dat = load("../experiments/exp2/data_exp2.jld2")
xobsT = dat["xobsT"]
#xobs0 = dat["xobs0"]

template_estimation(xobsT;ITER=150)
