module BridgeLandmarks

using Bridge
using ForwardDiff
using Plots

using LinearAlgebra

const d = 2

export Point, PointF, Unc, UncF

include("nstate.jl")
include("state.jl")
include("models.jl")
include("patches.jl")
#include("plotlandmarks.jl")  # keep, but presently unused as all is transferred to plotting in R
include("plotting.jl")
include("lmguid.jl")  # replacing lmguiding_mv and update_initialstate


end # module
