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

Random.seed!(9)

workdir = @__DIR__
cd(workdir)
include(joinpath(BL.dir(),"scripts", "postprocessing.jl"))
#mkdir(joinpath(workdir,"out"))
outdir = joinpath(workdir,"out")


if false # make example datasets with configurations
    dat = load("../experiments/exp1/data_exp1.jld2")
    xobs0 = dat["xobs0"]
    xobsT = dat["xobsT"]
    writedlm("landmarks0.txt", hcat(extractcomp(xobs0,1), extractcomp(xobs0,2)))
    writedlm("landmarksT.txt", hcat(extractcomp(xobsT,1), extractcomp(xobsT,2)))
end

# pass matrices to main function
landmarks0 = readdlm("landmarks0.txt")
landmarksT = readdlm("landmarksT.txt")


BL.landmarkmatching(landmarks0, landmarksT; outdir=outdir, ITER=10)

# @enter landmarkmatching(landmarks0, landmarksT)
# Juno.@enter landmarkmatching(landmarks0, landmarksT)

if false
    using BridgeLandmarks
    F = [(i==j) * one(UncF) for i in 1:5, j in 1:3]  # pick position indices
    F
    struct Test
        F
    end

    obj = Test(F)
    obj
    show(obj.F)
    show(obj)
    import Base.show
    function show(io::IO, mime::MIME"text/plain",obj::Test)
        print(io,mime, obj.F)
    end

    function show(io::IO, obj::Test)
        show(io,obj.F)
    end
    show(obj)
end
