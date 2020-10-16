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
workdir = @__DIR__
cd(workdir)
outdir = workdir

# read data
d1 = readdlm("dataset1.csv",',')
d2 = readdlm("dataset2.csv",',')
n = 12

# center and scale
center = PointF(40.0, 20.0)
xobs0 = map(i->PointF(d1[i,:]) - center, 1:n)/10.0
xobsT = map(i->PointF(d2[i,:]) - center, 1:n)/10.0

# set pars
p_ms = Pars_ms(δmom=0.01, σobs = 0.01)
# for ahs adjust domain bounds
p_ahs = Pars_ahs(δmom=0.01,db=[3.0,2.0],stdev=.5)


ups = [:innov, :mala_mom, :parameter]
ups = [:innov, :rmrw_mom, :parameter]#, :matching]

# run algorithm
@time landmarksmatching(xobs0,xobsT; ITER=200,pars=p_ms, outdir=outdir, updatescheme=ups)
#@time landmarksmatching(xobs0,xobsT; ITER=130,pars=p_ahs, outdir=outdir,  updatescheme=ups)


plotlandmarksmatching(outdir)
