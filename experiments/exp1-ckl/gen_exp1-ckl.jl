using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD2

workdir = @__DIR__
cd(workdir)

n = 29
nshapes = 1
T = 1.0

xobs0 = [PointF(2.0cos(t), sin(t))  for t in collect(0:(2pi/n):2pi)[2:end]]
# peach_xcoord(s) = (2.0 + sin(s)^3) * cos(s)
# peach_ycoord(s) = (2.0 + sin(s)^3) * sin(s)
# qT = [PointF(peach_xcoord(t), peach_ycoord(t))  for t in (0:(2pi/n):2pi)[1:n]]
heart_xcoord(s) = 0.2*(13cos(s)-5cos(2s)-2cos(3s)-cos(4s))
heart_ycoord(s) = 0.2*16(sin(s)^3)
xobsT = [PointF(heart_xcoord(t), heart_ycoord(t))  for t in (0:(2pi/n):2pi)[1:n]]

JLD2.@save "data_exp1-ckl.jld2" xobs0 xobsT n nshapes
