# ellipse to corpus callosum like shape
# initial and final landmark positions observed

using Revise
using BridgeLandmarks
const BL = BridgeLandmarks
using Distributions
using LinearAlgebra
using Random
using FileIO

# using RCall
# using Random
# using DataFrames
# using DelimitedFiles
# using CSV
# using StaticArrays
#using JLD2

Random.seed!(9)

workdir = @__DIR__
cd(workdir)
outdir = workdir

dat = load("data_exp-dgs.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]


p_ms = Pars_ms(δmom=0.01/n, cinit=1.0, γinit=1.0/√n, σobs = 0.01)
# for ahs adjust domain bounds
p_ahs = Pars_ahs(δmom=0.01/n, cinit=1.0, γinit=1.0/√n, db=[2.0,2.0],stdev=.25, σobs = 0.01)



ainit = 0.5*mean(norm.(diff(xobs0)))
ups = [:innov, :mala_mom]
ups = [:innov, :rmrw_mom, :parameter]

@time landmarksmatching(xobs0,xobsT; ITER=350,pars=p_ms, outdir=outdir, ainit=ainit, updatescheme=ups)
@time landmarksmatching(xobs0,xobsT; ITER=50,pars=p_ahs, outdir=outdir, ainit=ainit, updatescheme=ups)

plotlandmarksmatching(outdir)
