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
using DelimitedFiles
using CSV
using RCall

# using TimerOutputs
# const to = TimerOutput()

dir() = joinpath(@__DIR__, "..")

const d = 2
const sk = 1  # entries to skip for likelihood evaluation
const itostrat = true

import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

export Point, PointF, Unc, UncF, State, deepvec

export Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux, Pars_ms, Pars_ahs, FlatPrior
export landmarksforward, itostrat, construct_nfs, lm_mcmc, gramkernel, landmarksmatching, template_estimation, plotlandmarksmatching

export Lmplotbounds, extractcomp, tc

export d, sk, to

plotlandmarkpositions = Ref{Any}((args...) -> nothing )

include("nstate.jl")
include("state.jl")
include("models.jl")
include("patches.jl")
#include("plotlandmarks.jl")  # keep, but presently unused as all is transferred to plotting in R
include("plotting.jl")

include("pars.jl")  # set tuning pars
include("obsinfo.jl") # set observation info
include("guidedproposal.jl")
include("backwardsfiltering.jl")
include("lmguid.jl")  # contains main routines for mcmc
include("postprocessing.jl")
include("updatematching.jl")

include("landmarksmatching.jl")
include("template_estimation.jl")
include("plots_landmarksmatching.jl")
#include("updatematching.jl")

end # module
