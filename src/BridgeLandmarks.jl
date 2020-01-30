module BridgeLandmarks

using Bridge
using ForwardDiff
using Plots
using RecursiveArrayTools
using DataFrames
using Distributions
using GaussianDistributions
using SparseArrays
using StaticArrays

using LinearAlgebra, Base.Iterators
using PDMats
using JLD

using TimerOutputs
const to = TimerOutput()

dir() = joinpath(@__DIR__, "..")

const _d = 2
const sk = 1  # entries to skip for likelihood evaluation
const itostrat = true

export Point, PointF, Unc, UncF, State, deepvec
export point, pointf, unc, uncf

export Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux
export landmarksforward, itostrat, construct_nfs, lm_mcmc, gramkernel, tuningpars_mcmc

export Lmplotbounds, extractcomp, tc

export _d, sk, to
function dim(A::AbstractArray{T})  where {T<:StaticArray}
    @assert _d == size(T,1)
    size(T,1)
end
import Bridge: outer, inner

plotlandmarkpositions = Ref{Any}((args...) -> nothing )

include("nstate.jl")
include("state.jl")
include("models.jl")
include("patches.jl")
#include("plotlandmarks.jl")  # keep, but presently unused as all is transferred to plotting in R
include("plotting.jl")
include("lmguid.jl")  # replacing lmguiding_mv and update_initialstate
include("updatematching.jl")

end # module
