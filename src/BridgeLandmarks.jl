module BridgeLandmarks

using Bridge
using ForwardDiff
using Plots
using RecursiveArrayTools
using DataFrames
using Distributions
using GaussianDistributions
using SparseArrays

using LinearAlgebra, Base.Iterators
using PDMats

const d = 2
const sk=1  # entries to skip for likelihood evaluation
const itostrat = true

export Point, PointF, Unc, UncF, State, deepvec

export Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux
export landmarksforward, itostrat, construct_nfs, lm_mcmc

export Lmplotbounds, extractcomp, tc

export d, sk

plotlandmarkpositions = Ref{Any}((args...) -> nothing )

include("nstate.jl")
include("state.jl")
include("models.jl")
include("patches.jl")
#include("plotlandmarks.jl")  # keep, but presently unused as all is transferred to plotting in R
include("plotting.jl")
include("lmguid.jl")  # replacing lmguiding_mv and update_initialstate


end # module
