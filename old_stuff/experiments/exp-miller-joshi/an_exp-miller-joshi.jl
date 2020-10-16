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
using DelimitedFiles
using Distances


Random.seed!(9)
workdir = @__DIR__
cd(workdir)
outdir = workdir


d1 = readdlm("dataset1.csv",',')
d2 = readdlm("dataset2.csv",',')
n = 12

center = PointF(40.0, 20.0)
xobs0 = map(i->PointF(d1[i,:]) - center, 1:n)
xobsT = map(i->PointF(d2[i,:]) - center, 1:n)

# read data
# dat = load("data_exp1.jld2")
# xobs0 = dat["xobs0"]
# xobsT = dat["xobsT"]

# set pars
p_ms = Pars_ms(δmom=0.1, cinit=0.2, γinit=2.0, σobs = 0.01)

# for ahs adjust domain bounds
#p_ahs = Pars_ahs(δmom=0.1, cinit=0.02, γinit=0.2, db=[30.0,25.0],stdev=5.0, σobs = 0.01)
ainit = 0.5*mean(norm.(diff(xobs0)))  #pairwise(dist, X, Y, dims=2)
ups = [:innov, :mala_mom, :parameter]#, :matching]
ups = [:innov, :rmrw_mom, :parameter]#, :matching]

# run algorithm
@time landmarksmatching(xobs0,xobsT; ITER=200,pars=p_ms, outdir=outdir, ainit=ainit, updatescheme=ups)
#@time landmarksmatching(xobs0,xobsT; ITER=25,pars=p_ahs, outdir=outdir, ainit=ainit, updatescheme=ups)

plotlandmarksmatching(outdir)
