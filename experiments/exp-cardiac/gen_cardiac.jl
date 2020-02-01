# ellipse to multiple forward simulated shapes
# only final configurations observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD2
using NPZ

workdir = @__DIR__
cd(workdir)

cardiac = npzread("cardiac.npy")  # heart data (left ventricles, the one we used in https://arxiv.org/abs/1705.10943
cardiacx = cardiac[:,:,1]  # x-coordinates of landmarks
cardiacy = cardiac[:,:,2]  # y-coordinates of landmarks

landmarksset = 1:5:66
nshapes = 14

n = length(landmarksset)
xobsT = fill(zeros(PointF,66),nshapes)
for i in 1:nshapes # for each image
    xobsT[i] = [PointF(cardiacx[i,j], cardiacy[i,j]) for j in landmarksset ]
end

obs_atzero = false
xobs0 = 0

JLD2.@save "data_cardiac.jld2" xobs0 xobsT n nshapes
