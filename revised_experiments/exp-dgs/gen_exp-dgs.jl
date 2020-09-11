# dgs = data generated stefan
# only final landmark positions observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD2
using NPZ

workdir = @__DIR__
cd(workdir)

nshapes = 1

testshapes = npzread("match.npy.npz")
xobs0vec =  get(testshapes,"q0",0)
xobsTvec =  get(testshapes,"v",0)
nb = div(length(xobs0vec),2)

subs = 1:6:nb
xobs0 = [PointF(xobs0vec[2i-1],xobs0vec[2i]) for i in subs]
xobsT = [PointF(xobsTvec[2i-1],xobsTvec[2i]) for i in subs]
n = length(subs)

JLD2.@save "data_exp-dgs.jld2" xobs0 xobsT n nshapes
