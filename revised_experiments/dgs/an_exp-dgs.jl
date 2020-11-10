using Revise
using BridgeLandmarks
const BL = BridgeLandmarks
using Distributions
using LinearAlgebra
using Random
using FileIO
using JLD2
using DelimitedFiles


Random.seed!(9)

################ set directories ##########################################
workdir = @__DIR__
cd(workdir)
include("../../outdirpath.jl")

dir1 = joinpath(outdirpath,"exp-dgs/ms")
outdir_ms = mkpath(dir1)

dir2 = joinpath(outdirpath,"exp-dgs/ahs")
outdir_ahs = mkpath(dir2)

################ read data ##########################################
dat = load("data_exp-dgs.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = length(xobs0)


################ settings and mcmc #################################
ups = [:innov, :mala_mom, :parameter]
adaptskip = 100
skip_saveITER = 10
printskip = 1000
ITER = 300#2_000

p_ms = Pars_ms(γinit=1.0/√n, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01,
                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.9)
landmarksmatching(xobs0,xobsT; ITER=ITER, pars=p_ms, updatescheme=ups, printskip=printskip, outdir=outdir_ms)

p_ahs = Pars_ahs(db=[2.0, 2.0],stdev=.5,γinit=.1, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01,
                                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.9)
landmarksmatching(xobs0,xobsT; ITER=ITER, pars=p_ahs, updatescheme=ups, printskip=printskip, outdir=outdir_ahs)




## old (this worked ok)
# set pars
#p_ms = Pars_ms(δmom=0.001, σobs = 0.01)
# for ahs adjust domain bounds
#p_ahs = Pars_ahs(δmom=0.01,db=[2.0,2.0],stdev=.25)
