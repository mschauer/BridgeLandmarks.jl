# important: set d=1 in BridgeLandmarks.jl for this example
using Revise
using BridgeLandmarks
const BL=BridgeLandmarks
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

################ set directories ##########################################
workdir = @__DIR__
cd(workdir)
include("../../outdirpath.jl")

outdir_ms = joinpath(outdirpath,"exp1_1D/ms")
mkpath(outdir_ms)

outdir_ahs = joinpath(outdirpath,"exp1_1D/ahs")
mkpath(outdir_ahs)

################ read data ##########################################
dat = load("data_exp1-1D.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
nshapes = dat["nshapes"]

################ settings and mcmc #################################
skip_saveITER = 1000
printskip = 1000
adaptskip = ITER = 20_000

ups = [:innov, :mala_mom]
p_ms = Pars_ms(γinit=1.0/√n, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.001, σobs=0.001,
                adaptskip=adaptskip, skip_saveITER=skip_saveITER)
landmarksmatching(xobs0,xobsT; ITER=ITER, pars=p_ms, updatescheme=ups, printskip=printskip, outdir=outdir_ms, ainit=1.0)


p_ahs = Pars_ahs(db=[2.5],stdev=.5, γinit=0.1, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.001, σobs=0.001,
                                  adaptskip=adaptskip, skip_saveITER=skip_saveITER)
landmarksmatching(xobs0,xobsT; ITER=ITER, pars=p_ahs, updatescheme=ups, printskip=printskip, outdir=outdir_ahs, ainit=1.0)
