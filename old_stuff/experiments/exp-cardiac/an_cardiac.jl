using Revise
using BridgeLandmarks
const BL = BridgeLandmarks
using Distributions
using LinearAlgebra
using Random
using FileIO

Random.seed!(9)

workdir = @__DIR__
cd(workdir)
outdir = workdir


#-------- read data ----------------------------------------------------------
dat = load("data_cardiac.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]


# set pars
p_ms = Pars_ms(δpos=0.00002, δmom=0.1, cinit=0.2, γinit=1.0, σobs = 0.01, ρinit=0.3)
p_ahs = Pars_ahs(δpos=0.01, δmom=0.1, cinit=0.05, γinit=0.5, db=[2.5,2.5],stdev=0.75, σobs = 0.01, ρinit=0.4)

ITER = 2000 # taken in paper
template_estimation(xobsT; ainit=0.1, xinitq=xobsT[1],
pars = p_ms, outdir=outdir, ITER=200, updatescheme = [:innov, :rmmala_pos, :parameter])  # deliberately initialise badly to show it works

plottemplate_estimation(outdir)
