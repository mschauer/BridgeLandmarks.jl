# ellipse to multiple forward simulated shapes
# only final configurations observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD
using NPZ

workdir = @__DIR__
cd(workdir)


cardiac = npzread("cardiac.npy")  # heart data (left ventricles, the one we used in https://arxiv.org/abs/1705.10943
cardiacx = cardiac[:,:,1]  # x-coordinates of landmarks
cardiacy = cardiac[:,:,2]  # y-coordinates of landmarks

landmarksset = 1:5:66
nshapes = length(landmarksset)

n = length(landmarksset)
xobsT = fill(zeros(PointF,66),nshapes)
for i in 1:nshapes # for each image
    xobsT[i] = [PointF(cardiacx[i,j], cardiacy[i,j]) for j in landmarksset ]
end
x0 = State(zeros(PointF,n), zeros(PointF,n))


obs_atzero = false
xobs0 = []

pb = Lmplotbounds(-0.25,0.25,-0.25,0.25)

if false # some simple visualisation of the data
     Xf = 0
    if model == :ms
        P = MarslandShardlow(P.a,P.c, P.γ, P.λ, n)
    elseif model== :ahs
        P = Landmarks(P.a,P.c, n, P.db, P.nfstd, P.nfs)
    end

    xobsTdf = DataFrame(x=extractcomp(xobsT[1],1), y =extractcomp(xobsT[1],2))
    @rput xobsTdf
    R"""
        library(tidyverse)
        library(ggplot2)
        xobsTdf %>% ggplot(aes(x=x,y=y)) + geom_point() + geom_path()
    """
end





save("data_cardiac.jld", "xobs0",xobs0, "xobsT", xobsT, "n", n, "x0", x0, "pb", pb, "nshapes", nshapes)
