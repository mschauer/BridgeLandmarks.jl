using Revise
using BridgeLandmarks
const BL = BridgeLandmarks
using Distributions
using LinearAlgebra
using Random
# using FileIO
# using JLD2
using DelimitedFiles


Random.seed!(9)

################ set directories ##########################################
workdir = @__DIR__
cd(workdir)
include("../../outdirpath.jl")

dir1 = joinpath(outdirpath,"millerjoshi/ms")
outdir_ms = mkpath(dir1)

dir2 = joinpath(outdirpath,"millerjoshi/ahs")
outdir_ahs = mkpath(dir2)

################ read data ##########################################
workdir = @__DIR__
cd(workdir)
d1 = readdlm("dataset1.csv",',')
d2 = readdlm("dataset2.csv",',')
n = 12
# center and scale
center = PointF(40.0, 20.0)
xobs0 = map(i->PointF(d1[i,:]) - center, 1:n)/10.0
xobsT = map(i->PointF(d2[i,:]) - center, 1:n)/10.0

################ settings and mcmc #################################
ups = [:innov, :mala_mom, :parameter]
adaptskip = 100
skip_saveITER = 100
printskip = 1000
ITER = 25_000

#ITER = 100
#p_ms = Pars_ms(γinit=1.0/√n, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01,
#                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.9)

using ProfileView
using Profile
#Profile.init() @profview landmarksmatching(xobs0,xobsT; ITER=ITER, pars=p_ms, updatescheme=ups, printskip=printskip, outdir=outdir_ms)




p_ahs = Pars_ahs(db=[3.0, 2.0],stdev=.5,γinit=.1, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01,
                                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.9)
ITER = 20
landmarksmatching(xobs0,xobsT; ITER=ITER, pars=p_ahs, updatescheme=ups, printskip=printskip, outdir=outdir_ahs)
ITER = 20
Profile.init(); @profile landmarksmatching(xobs0,xobsT; ITER=ITER, pars=p_ahs, updatescheme=ups, printskip=printskip, outdir=outdir_ahs)
ProfileView.view()
