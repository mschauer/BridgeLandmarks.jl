using StaticArrays

const Point{T} = SArray{Tuple{d},T,1,d}       # point in Rd
const Unc{T} = SArray{Tuple{d,d},T,2,d*d}     # Matrix presenting uncertainty
const PointF = Point{Float64}
const UncF = Unc{Float64}

struct NState{P, M<:AbstractMatrix{P}}
    x::M
end
const State = NState

NState(x::Vector) = NState(reshape(x, (2, length(x)>>1)))

import Base: axes, #=iterate,=# eltype, copy, copyto!, zero, eachindex, getindex, setindex!, size, vec

q(x::NState, i) = x.x[1, i]
p(x::NState, i) = x.x[2, i]
eltype(x::NState) = eltype(x.x)
deepeltype(x::Union{NState,Vector}) = eltype(eltype(x))
q(x::NState) = @view x.x[1:2:end]
p(x::NState) = @view x.x[2:2:end]

import LinearAlgebra: norm
norm(x::NState) = norm(vec(x))

vecofpoints2state(x::Vector) = NState(x)

size(s::NState) = size(s.x)
axes(s::NState, i) = axes(s.x, i)
deepvec(x::NState) = vec(reinterpret(deepeltype(x), x.x))
deepvec(x::Vector) = reinterpret(deepeltype(x), x)

function deepvec2state(x::Vector)
    x = reinterpret(Point{eltype(x)}, x)
    #dump(length(x))
    NState(reshape(x, (2, length(x)>>1)))
end

function deepvec2state(x::Base.ReshapedArray{<:Any,1,<:Base.ReinterpretArray})
    NState(x.parent.parent)
end

function NState(q::AbstractVector, p::AbstractVector)
    length(p) != length(q) && throw(DimensionMismatch())
    NState([((q,p)[i])[j] for i in 1:2, j in 1:length(p)])
end

vec(x::NState) = vec(x.x)
Base.broadcastable(x::NState) = x
Broadcast.BroadcastStyle(::Type{<:NState}) = Broadcast.Style{NState}()



function Base.getproperty(u::NState, s::Symbol)
    x = getfield(u, :x)
    s == :x && return x

    if s == :q
        @view x[1:2:end]
    elseif s == :p
        @view x[2:2:end]
    else
        throw(ArgumentError("NState has properties `p`, `q` or `x`"))
    end
end

zero!(v::NState) = zero!(v.x)
zero!(v::AbstractArray) = v .= Ref(zero(eltype(v)))
zero(v::NState) = NState(zero(v.x))

copy(x::NState) = NState(copy(x.x))
function copyto!(x::NState, y::NState)
    copyto!(x.x, y.x)
    x
end

function copyto!(x::NState, y)
    for i in eachindex(x)
        x[i] = y[i]
    end
    x
end

getindex(x::NState, I) = getindex(x.x, I)
function setindex!(x::NState, val, I)
    x.x[I] = val
end

eachindex(x::NState) = CartesianIndices(axes(x.x))

import Base: *, +, /, -
import LinearAlgebra: dot

#    import Bridge: outer, inner

function outer(x::NState, y::NState)
    [outer(x[i],y[j]) for i in eachindex(x), j in eachindex(y)]
end

*(c::Number, x::NState) = NState(c*x.x)
*(x::NState, c::Number) = NState(x.x*c)
+(x::NState, y::NState) = NState(x.x + y.x)
-(x::NState, y::NState) = NState(x.x - y.x)
/(x::NState, y) = NState(x.x/y)

dot(x::NState, y::NState) = dot(vec(x),vec(y))

function Base.:*(A::AbstractArray{<:Unc,2}, x::State)
    vecofpoints2state(A*vec(x))
end


q(i::Int) = 2i - 1
p(i::Int) = 2i

flipmomenta(x::NState) = NState(x.q, -x.p)

onemask(x) = onemask.(x)

"""
    onemask(x::Number) = one(x)
"""
onemask(x::Number) = one(x)
