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
p_ms = Pars_ms(δmom=0.01/n, cinit=1.0, γinit=1.0/√n, σobs = 0.01)
# for ahs adjust domain bounds
p_ahs = Pars_ahs(δmom=0.01, cinit=1.0, γinit=1.0/√n, db=[3.0,2.0],stdev=0.5, σobs = 0.1)

pars=p_ahs; construct_nfs(pars.db, pars.stdev, pars.γinit)

ainit = 0.5*mean(norm.(diff(xobs0)))
ups = [:innov, :mala_mom, :parameter]
ups = [:innov, :rmrw_mom, :parameter]#, :matching]
ups = [:rmrw_mom, :parameter]

# run algorithm
@time landmarksmatching(xobs0,xobsT; ITER=100,pars=p_ms, outdir=outdir, ainit=ainit, updatescheme=ups)
@time landmarksmatching(xobs0,xobsT; ITER=130,pars=p_ahs, outdir=outdir, ainit=ainit, updatescheme=ups)


plotlandmarksmatching(outdir)
