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
include("outdirpath.jl")

dir1 = joinpath(outdirpath,"millerjoshi/ms")
outdir_ms = mkpath(dir1)

dir2 = joinpath(outdirpath,"millerjoshi/ahs")
outdir_ahs = mkpath(dir2)

################ read data ##########################################
workdir = @__DIR__
cd(workdir)
d1 = readdlm("revised_experiments/millerjoshi/dataset1.csv",',')
d2 = readdlm("revised_experiments/millerjoshi/dataset2.csv",',')
n = 12
# center and scale
center = PointF(40.0, 20.0)
xobs0 = map(i->PointF(d1[i,:]) - center, 1:n)/10.0
xobsT = map(i->PointF(d2[i,:]) - center, 1:n)/10.0

################ settings and mcmc #################################
ups = [:innov, :mala_mom, :parameter]
skip_saveITER = 1000
printskip = 1000
adaptskip = ITER = 20_000

p_ms = Pars_ms(γinit=1.0/√n, aprior=Pareto(1.0, 0.1), dt = 0.01,
                adaptskip=adaptskip, skip_saveITER=skip_saveITER,
                ρlowerbound=0.9)

@time landmarksmatching(xobs0,xobsT; ITER=ITER, pars=p_ms, updatescheme=ups, printskip=printskip, outdir=outdir_ms)

p_ahs = Pars_ahs(db=[3.0, 2.0], stdev=.5, γinit=.1, aprior=Pareto(1.0, 0.1), dt = 0.01,
                                adaptskip=adaptskip, skip_saveITER=skip_saveITER,
                                ρlowerbound=0.9)
@time landmarksmatching(xobs0,xobsT; ITER=ITER, pars=p_ahs, updatescheme=ups, printskip=printskip, outdir=outdir_ahs)



# using ProfileView
# using Profile
#Profile.init() @profview
