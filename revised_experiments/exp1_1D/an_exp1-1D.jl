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

dir1 = joinpath(outdirpath,"exp1_1D/ms")
outdir_ms = mkpath(dir1)

dir2 = joinpath(outdirpath,"exp1_1D/ahs")
outdir_ahs = mkpath(dir2)

################ read data ##########################################
dat = load("data_exp1-1D.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]
nshapes = dat["nshapes"]

################ settings and mcmc #################################
ups = [:innov, :mala_mom, :parameter]
adaptskip = 100
skip_saveITER = 100
printskip = 1000
ITER = 50_000

#p_ms = Pars_ms(δmom=10e-2, γinit=1.0/√n, aprior=Pareto(1.0, 0.1),
#                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρinit = 0.99, σobs=0.01)#, δa=0.0) #,δa = 0.2, δγ=0.2)
ups = [:innov, :mala_mom]
#ups = [:innov, :rmrw_mom]
p_ms = Pars_ms(γinit=1.0/√n, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01,
                                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρinit = 0.99, σobs=0.01)#, δa=0.0) #,δa = 0.2, δγ=0.2)
ups = [:mala_mom, :innov]
 ups = [:mala_mom]
# ups = [:innov]
ups = [:rmrw_mom]
landmarksmatching(xobs0, xobsT; ITER=ITER, pars=p_ms, updatescheme=ups, printskip=printskip, outdir=outdir_ms, ainit=1.0)



ups = [:innov, :rmrw_mom, :parameter]
p_ahs = Pars_ahs(δmom=10e-2,db=[2.0],stdev=.5,γinit=1.0, aprior=Pareto(1.0, 0.1),
                adaptskip=adaptskip, skip_saveITER=skip_saveITER)
landmarksmatching(xobs0,xobsT; ITER=ITER,pars=p_ahs, updatescheme=ups, printskip=printskip, outdir=outdir_ahs)

#plotlandmarksmatching(outdir)
