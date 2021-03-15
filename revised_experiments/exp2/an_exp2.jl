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

dir1 = joinpath(outdirpath,"exp2/ms")
outdir_ms = mkpath(dir1)

dir2 = joinpath(outdirpath,"exp2/ahs")
outdir_ahs = mkpath(dir2)

################ read data ##########################################
dat = load("data_exp2.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]

################ settings and mcmc #################################
ups =  [:innov, :rmmala_pos, :parameter]
adaptskip = 100
skip_saveITER = 50#0
printskip = 25
ITER = 2500#0



δpos = (d*n)^(-1/6) * [0.001, 0.0001]
p_ms = Pars_ms(δpos=δpos,  γinit=.1/√n,  ## LET OP
                  aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.01,
                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.8, δa=0.02)
p_ahs = Pars_ahs(δpos=δpos,  db=[2.5, 2.5],stdev=.25,γinit=.1, aprior=Pareto(1.0, 0.1), η =  n -> 0.0, dt = 0.001,
                                adaptskip=adaptskip, skip_saveITER=skip_saveITER, ρlowerbound=0.9)
# original experiment
#p_ms = Pars_ms(ρinit = 0.7,covθprop = [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04], δpos=0.005, δmom=0.1,cinit=0.2,γinit=2.0  )
#p_ahs = Pars_ahs(ρinit = 0.7,covθprop = [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04], δpos=0.01, δmom=0.1,cinit=0.02,γinit=2.0, stdev=0.75, db = [2.5, 2.5]  )

################## deliberately start in a wrong configuration ###################
n = length(xobsT[1])
xinitq = xobsT[1]
θ, ψ =  π/6, 0.25
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
xinitq_adj = [rot * stretch * xinitq[i] for i in 1:n ]

template_estimation(xobsT; xinitq=xinitq_adj,pars = p_ms,
     ITER=ITER, updatescheme = ups, printskip=printskip, outdir=outdir_ms)

#template_estimation(xobsT; xinitq=xinitq_adj,pars = p_ahs,
#      ITER=ITER, updatescheme = ups, printskip=printskip, outdir=outdir_ahs)
#plottemplate_estimation(outdir)
