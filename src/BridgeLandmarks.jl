module BridgeLandmarks

using Bridge
using ForwardDiff
using Plots
using RecursiveArrayTools
using DataFrames
using Distributions
using GaussianDistributions
using SparseArrays
using Parameters

using LinearAlgebra, Base.Iterators
using PDMats
using JLD

using TimerOutputs
const to = TimerOutput()

dir() = joinpath(@__DIR__, "..")

const d = 2
const sk = 1  # entries to skip for likelihood evaluation
const itostrat = true

export Point, PointF, Unc, UncF, State, deepvec

export Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux
export landmarksforward, itostrat, construct_nfs, lm_mcmc, gramkernel, MCMC_TuningPars

export Lmplotbounds, extractcomp, tc

export d, sk, to

plotlandmarkpositions = Ref{Any}((args...) -> nothing )

include("nstate.jl")
include("state.jl")
include("models.jl")
include("patches.jl")
#include("plotlandmarks.jl")  # keep, but presently unused as all is transferred to plotting in R
include("plotting.jl")
include("guidedproposal.jl")
include("lmguid.jl")
include("updatematching.jl")

end # module
