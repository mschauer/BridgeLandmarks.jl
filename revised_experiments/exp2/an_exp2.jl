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

workdir = @__DIR__
cd(workdir)
outdir = workdir

#-------- read data ----------------------------------------------------------
dat = load("data_exp2.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
#q0 = dat["q0"]

################################# start settings #################################


ups =  [:innov, :rmmala_pos, :parameter]

p_ms = Pars_ms(δmom=0.01, σobs = 0.01)
# for ahs adjust domain bounds
p_ahs = Pars_ahs(δmom=0.01,db=[2.5,1.5],stdev=.25)

# original experiment
#p_ms = Pars_ms(ρinit = 0.7,covθprop = [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04], δpos=0.005, δmom=0.1,cinit=0.2,γinit=2.0  )
#p_ahs = Pars_ahs(ρinit = 0.7,covθprop = [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04], δpos=0.01, δmom=0.1,cinit=0.02,γinit=2.0, stdev=0.75, db = [2.5, 2.5]  )


ainit = mean(norm.(diff(xobsT[1])))/2.0

n = length(xobsT[1])
xinitq = xobsT[1]
θ, ψ =  π/6, 0.25
rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
stretch = SMatrix{2,2}(1.0 + ψ, 0.0, 0.0, 1.0 - ψ)
xinitq_adj = [rot * stretch * xinitq[i] for i in 1:n ]


ITER = 200
template_estimation(xobsT; ainit=ainit, xinitq=xinitq_adj,
    pars = p_ms, outdir=outdir, ITER=100, updatescheme = ups, printskip=10)  # deliberately initialise badly to show it works

plottemplate_estimation(outdir)
