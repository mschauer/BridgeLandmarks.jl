# ellipse to rotated and shifted ellipse
# initial and final landmark positions observed
using Revise

using BridgeLandmarks
const BL = BridgeLandmarks
using RCall
using Plots
using Random
using Distributions
using DataFrames
using DelimitedFiles
using CSV
using StaticArrays
using LinearAlgebra
using JLD2
using FileIO



Random.seed!(9)

workdir = @__DIR__
cd(workdir)
outdir = workdir

dat = load("data_exp1-translated.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]


p_ms = Pars_ms(δmom=0.01, σobs = 0.01)
# for ahs adjust domain bounds
p_ahs = Pars_ahs(δmom=0.01,db=[2.5,1.5],stdev=.25)


ainit =  0.5*mean(norm.(diff(xobs0)))
ups = [:innov, :mala_mom, :parameter]


@time landmarksmatching(xobs0,xobsT; ITER=10_000,pars=p_ms, outdir=outdir, ainit=ainit, updatescheme=ups)
@time landmarksmatching(xobs0,xobsT; ITER=500,pars=p_ahs, outdir=outdir, ainit=ainit, updatescheme=ups)

plotlandmarksmatching(outdir)
