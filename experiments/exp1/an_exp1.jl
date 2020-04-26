# ellipse to rotated and shifted ellipse, initial and final landmark positions observed
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

# read data
dat = load("data_exp1.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]

# set pars
p_ms = Pars_ms(δmom=0.1, cinit=0.2, γinit=2.0, σobs = 0.01)
p_ahs = Pars_ahs(δmom=0.1, cinit=0.02, γinit=0.2, db=[2.5,2.5],stdev=0.75, σobs = 0.01)
ainit = mean(norm.(diff(xobs0)))/2.0
ups = [:innov, :mala_mom]

# run algorithm
@time landmarksmatching(xobs0,xobsT; ITER=500,pars=p_ms, outdir=outdir, ainit=ainit, updatescheme=ups)
@time landmarksmatching(xobs0,xobsT; ITER=500,pars=p_ahs, outdir=outdir, ainit=ainit, updatescheme=ups)

plotlandmarksmatching(outdir)
