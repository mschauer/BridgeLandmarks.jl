# ellipse to multiple forward simulated shapes
# only final landmark positions observed

using BridgeLandmarks
using Random
using Distributions
using StaticArrays
using LinearAlgebra
using JLD

workdir = @__DIR__
cd(workdir)

n = 40
nshapes = 1
T = 1.0

q0 = [PointF(2.0cos(t), 2.0sin(t))  for t in collect(0:(2pi/n):2pi)[2:end]]
p0 = zeros(PointF,n)
x0 = State(q0, p0)
xobs0 = x0.q
# peach_xcoord(s) = (2.0 + sin(s)^3) * cos(s)
# peach_ycoord(s) = (2.0 + sin(s)^3) * sin(s)
# qT = [PointF(peach_xcoord(t), peach_ycoord(t))  for t in (0:(2pi/n):2pi)[1:n]]
heart_xcoord(s) = 0.2*(13cos(s)-5cos(2s)-2cos(3s)-cos(4s))
heart_ycoord(s) = 0.2*16(sin(s)^3)
qT = [PointF(heart_xcoord(t), heart_ycoord(t))  for t in (0:(2pi/n):2pi)[1:n]]

xobsT = qT
pb = Lmplotbounds(-2.0,2.0,-1.5,1.5)



save("data_exp1-ckl.jld", "xobs0",xobs0, "xobsT", xobsT, "n", n, "x0", x0, "pb", pb, "nshapes", nshapes)
