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
# p_ms = Pars_ms(δpos=0.00002, δmom=0.1, cinit=0.2, γinit=1.0, σobs = 0.01, ρinit=0.3)
# p_ahs = Pars_ahs(δpos=0.01, δmom=0.1, cinit=0.05, γinit=0.5, db=[2.5,2.5],stdev=0.75, σobs = 0.01, ρinit=0.4)

p_ms = Pars_ms(δmom=0.01, σobs = 0.01,δpos=1.0e-6, δa=0.01)
# for ahs adjust domain bounds
#p_ahs = Pars_ahs(δmom=0.01,db=[2.5,1.5],stdev=.25)

ups =  [:innov, :rmmala_pos, :parameter]

template_estimation(xobsT; ainit=0.1, xinitq=xobsT[1],
    pars = p_ms, outdir=outdir, ITER=50, updatescheme = ups, printskip=10)  # deliberately initialise badly to show it works


plottemplate_estimation(outdir)
