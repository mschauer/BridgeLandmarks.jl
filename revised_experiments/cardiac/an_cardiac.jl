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

dir1 = joinpath(outdirpath,"cardiac/ms")
outdir_ms = mkpath(dir1)

dir2 = joinpath(outdirpath,"cardiac/ahs")
outdir_ahs = mkpath(dir2)

################ read data ##########################################
dat = load("data_cardiac.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]

################ settings and mcmc #################################
ups =  [:innov, :rmmala_pos, :parameter]
adaptskip = 100
skip_saveITER = 200
printskip = 25
ITER = 10_000



δpos = (d*n)^(-1/6) * [0.01, 0.001, 0.0001]
p_ms = Pars_ms(δpos=δpos,  γinit=.1/√n,  ## LET OP
                  aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01,
                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.9)
p_ahs = Pars_ahs(δpos=δpos,  db=[2.5, 2.5],stdev=.25,γinit=.1, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.001,
                                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.9)

template_estimation(xobsT; xinitq=xinitq_adj,pars = p_ms, ITER=ITER, updatescheme = ups, printskip=printskip, outdir=outdir_ms)

template_estimation(xobsT; xinitq=xinitq_adj,pars = p_ahs, ITER=ITER, updatescheme = ups, printskip=printskip, outdir=outdir_ahs)


# from 'old experiments'
#p_ms = Pars_ms(δmom=0.01, σobs = 0.01,δpos=1.0e-6, δa=0.01)
# for ahs adjust domain bounds
#p_ahs = Pars_ahs(δmom=0.01,db=[2.5,1.5],stdev=.25)
