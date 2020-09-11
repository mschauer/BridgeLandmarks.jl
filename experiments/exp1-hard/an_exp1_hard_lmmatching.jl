# ellipse to rotated and shifted ellipse, initial and final landmark positions observed
using Revise
using Polynomials.PolyCompat
using BridgeLandmarks
const BL = BridgeLandmarks
using Distributions
using LinearAlgebra
using Random
using FileIO
using JLD2

Random.seed!(9)
workdir = @__DIR__
cd(workdir)
outdir = workdir

# read data
dat = load("data_exp1_hard.jld2")
xobs0 = dat["xobs0"]
xobsT = dat["xobsT"]
n = dat["n"]

# set pars
p_ms = Pars_ms(δmom=0.01/n, cinit=0.2, γinit=2.0/√n, σobs = 0.001, skip_saveITER=1)
#p_ahs = Pars_ahs(δmom=0.1, cinit=0.02, γinit=0.2, db=[2.5,2.5],stdev=0.75, σobs = 0.01, skip_saveITER=1)
ainit = mean(norm.(diff(xobs0)))/1.0
#ups = [:rmrw_mom, :innov, :parameter]
#ups = [:sgd_mom]#, :innov, :parameter]
#ups = [:rmrw_mom, :parameter, :innov]
ups = [:mala_mom, :parameter, :innov]

# run algorithm
@time landmarksmatching(xobs0,xobsT; ITER=10,pars=p_ms, outdir=outdir, ainit=ainit, updatescheme=ups)
#@time landmarksmatching(xobs0,xobsT; ITER=10,pars=p_ahs, outdir=outdir, ainit=ainit, updatescheme=ups)

plotlandmarksmatching(outdir)
