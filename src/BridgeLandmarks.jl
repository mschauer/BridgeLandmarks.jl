module BridgeLandmarks

using Bridge
using CSV
using DataFrames
using DelimitedFiles
using Distributions
using ForwardDiff
using GaussianDistributions
using JLD2
using LinearAlgebra
using Base.Iterators
using PDMats
using Parameters
using RecursiveArrayTools
using SparseArrays
using StaticArrays
using StatsFuns

dir() = joinpath(@__DIR__, "..")

const d = 2   # dimension of landmarks
const sk = 1  # entries to skip for likelihood evaluation
const itostrat =  true # include Ito-Stratonovich correction in AHS-model
const γconstant = true # if true, no parameter updating for γ

import Bridge: kernelr3!, R3!, target, auxiliary, constdiff, llikelihood, _b!, B!, σ!, b!

export Point, PointF, Unc, UncF, State, deepvec

export Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux, Pars_ms, Pars_ahs, FlatPrior, show_updates
export landmarksforward, itostrat, construct_nfs, lm_mcmc, gramkernel, landmarksmatching, template_estimation#, plotlandmarksmatching, plottemplate_estimation
export Lmplotbounds, extractcomp, tc
export d, sk, to

#plotlandmarkpositions = Ref{Any}((args...) -> nothing )

include("nstate.jl")
include("state.jl")

include("backwardsfiltering.jl")
include("guidedproposal.jl")
include("landmarksmatching.jl")
include("lmguid.jl")  # contains main routines for mcmc
include("models.jl")

include("obsinfo.jl") # set observation info
include("pars.jl")  # set tuning pars
include("patches.jl")
include("postprocessing.jl")

include("template_estimation.jl")
include("updatematching.jl")












# include("basic_outplots.jl")
# include("plotting.jl")


end # module
