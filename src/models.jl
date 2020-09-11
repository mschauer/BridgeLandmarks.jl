#### Landmarks specification
import Bridge: _b, _b!, B!, σ!, b!, σ, b, auxiliary

"""
    MarslandShardlow{T} <: ContinuousTimeProcess{State{PointF}}

## Arguments
- `a`: Hamiltonian kernel parameter
- `c`: kernel multiplicate parameter
- `γ`:  noise level
- `λ`:  mean reversion parameter (heath-bath parameter in Marsland-Shardlow (2017))
-  `n`: number of landmarks
"""
struct MarslandShardlow{T} <: ContinuousTimeProcess{State{PointF}}
    a::T # kernel std parameter
    c::T # kernel multiplicate parameter
    γ::T # noise level
    λ::T # mean reversion
    n::Int
end

"""
    MarslandShardlowAux{S,T} <: ContinuousTimeProcess{State{PointF}}

## Arguments
- `a`: Hamiltonian kernel parameter
- `c`: kernel multiplicate parameter
- `γ`:  noise level
- `λ`:  mean reversion parameter (heath-bath parameter in Marsland-Shardlow (2017))
- `xT`: State at time T used for constructing the auxiliary process
-  `n`: number of landmarks
"""
struct MarslandShardlowAux{S,T} <: ContinuousTimeProcess{State{PointF}}
    a::T # kernel std parameter
    c::T # kernel multiplicate parameter
    γ::T # noise level
    λ::T # mean reversion
    xT::State{Point{S}}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int
end

struct Noisefield{T}
    δ::Point{T}   # locations of noise field
    γ::Point{T}  # scaling at noise field (used to be called lambda)
    τ::T # std of Gaussian kernel noise field
end

"""
    Landmarks{S,T} <: ContinuousTimeProcess{State{PointF}}

## Arguments
- `a`: Hamiltonian kernel parameter
- `c`: kernel multiplicate parameter
-  `n`:: Int64 number of landmarks
- `db`::Float64 square domain bound used for construction of noise fields
- `nfstd`:  standard deviation of noisefields (assumed to be the same for all noisefields)
- `nfs`:  vector of noisefields
"""
struct  Landmarks{S,T} <: ContinuousTimeProcess{State{PointF}}
    a::T # kernel std
    c::T # kernel multiplicate parameter
    n::Int64   # numer of landmarks
    db::Array{Float64,1} # domainbound
    nfstd::Float64
    nfs::Vector{Noisefield{S}}  # vector containing pars of noisefields
end

"""
    LandmarksAux{S,T} <: ContinuousTimeProcess{State{PointF}}

## Arguments
- `a`: Hamiltonian kernel parameter
- `c`: kernel multiplicate parameter
- `xT`: State at time T used for constructing the auxiliary process
-  `n`: number of landmarks
- `nfs`:  vector of noisefields
"""
struct LandmarksAux{S,T} <: ContinuousTimeProcess{State{PointF}}
    a::T # kernel std
    c::T # kernel multiplicate parameter
    xT::State{Point{S}}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int64   # numer of landmarks
    nfs::Vector{Noisefield{S}}  # vector containing pars of noisefields
end

"""
    MarslandShardlowAux(P::MarslandShardlow, xT) = MarslandShardlowAux(P.a,P.c, P.γ, P.λ, xT, P.n)
"""
MarslandShardlowAux(P::MarslandShardlow, xT) = MarslandShardlowAux(P.a,P.c, P.γ, P.λ, xT, P.n)

"""
    LandmarksAux(P::Landmarks, xT) = LandmarksAux(P.a,P.c, xT, P.n, P.nfs)
"""
LandmarksAux(P::Landmarks, xT) = LandmarksAux(P.a,P.c, xT, P.n, P.nfs)

"""
    auxiliary(P::Union{MarslandShardlow, Landmarks}, xT)

Construct auxiliary process corresponding to `P` and `xT`
"""
function auxiliary(P::Union{MarslandShardlow, Landmarks}, xT)
    if isa(P,MarslandShardlow)
        return MarslandShardlowAux(P,xT)
    elseif isa(P,Landmarks)
        return LandmarksAux(P,xT)
    end
end


const LandmarkModel = Union{Landmarks, LandmarksAux, MarslandShardlow, MarslandShardlowAux}

Bridge.constdiff(::Union{MarslandShardlow, MarslandShardlowAux,LandmarksAux}) = true
Bridge.constdiff(::Landmarks) = false


"""
    kernel(q, P::LandmarkModel)

kernel in Hamiltonian: P.c * exp(-Bridge.inner(q)/(2*P.a^2))
"""
function kernel(q, P::LandmarkModel)
   P.c * exp(-Bridge.inner(q)/(2*P.a^2))
end

"""
    ∇kernel(q, P::LandmarkModel)

gradient of kernel in hamiltonian
"""
function ∇kernel(q, P::LandmarkModel)
    -P.c * P.a^(-2) * kernel(q, P) * q
end

"""
Needed for b! in case P is auxiliary process
"""
function ∇kernel(q, qT, P::LandmarkModel)
     -P.a^(-2) *P.c  * kernel(qT, P) * q
end

"""
    hamiltonian(x, P)

Hamiltonian for deterministic part of landmarks model
"""
function hamiltonian(x, P)
    s = 0.0
    for i in axes(x, 2), j in axes(x, 2)
#        s += 1/2*dot(p[i], p[j])*kernel(q[i] - q[j], P)
        s += 1/2*dot(x.p[i], x.p[j])*kernel(x.q[i] - x.q[j], P)
    end
    s
end

Bridge.b(t::Float64, x, P::Union{Landmarks,MarslandShardlow})= Bridge.b!(t, x, copy(x), P)

Bridge.σ(t, x, dm, P) =  Bridge.σ!(t, x, dm , 0*x, P)

########################################################################################################################################################################################
################ MS model #########################################################################################


"""
    Bridge.b!(t, x, out, P::MarslandShardlow)

Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::MarslandShardlow)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += p(x,j)*kernel(q(x,i) - q(x,j), P)
            out.p[i] += -P.λ*p(x,j)*kernel(q(x,i) - q(x,j), P) -
                 dot(p(x,i), p(x,j)) * ∇kernel(q(x,i) - q(x,j), P)
        end
    end
    out
end

"""
    Bridge.b!(t, x, out, Paux::MarslandShardlowAux)

Evaluate drift of landmarks auxiliary process in (t,x) and save to out
x is a state and out as well
"""
# function Bridge.b!(t, x, out, Paux::MarslandShardlowAux)
#     zero!(out)
#     for i in 1:Paux.n
#         for j in 1:Paux.n
#             out.q[i] += p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
#             out.p[i] += -Paux.λ*p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
#         end
#     end
#     out
# end
function Bridge.b!(t, x, out, Paux::MarslandShardlowAux)
    zero!(out)
    for i in 1:Paux.n
        for j in 1:Paux.n
            out.q[i] += p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
        end
    end
    NState(out.q, - Paux.λ* out.q)
end



"""
    Bridge.B(t, Paux::MarslandShardlowAux)

Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::MarslandShardlowAux) # not AD safe
    X = zeros(UncF, 2Paux.n, 2Paux.n)
    for i in 1:Paux.n
        for j in 1:Paux.n
            X[2i-1,2j] =  kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux) * one(UncF)
            X[2i,2j] = -Paux.λ*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*one(UncF)
        end
    end
    X
end


"""
    Bridge.B!(t,X,out, Paux::MarslandShardlowAux)

Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::MarslandShardlowAux)
    out .= 0.0 * out
    for i in 1:Paux.n  # separately loop over even and odd indices
        for k in 1:2Paux.n # loop over all columns
            for j in 1:Paux.n
                 out[2i-1,k] += kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux) * X[p(j), k]
                 out[2i,k] += -Paux.λ*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux) * X[p(j), k]
            end
        end
    end
    out
end

function Bridge.β(t, Paux::MarslandShardlowAux) # Not AD save
    State(zeros(PointF,Paux.n), zeros(PointF,Paux.n))
end

"""
    Bridge.σ!(t, x, dm, out, P::Union{MarslandShardlow, MarslandShardlowAux})

Compute σ(t,x) * dm and write to out
"""
function Bridge.σ!(t, x, dm, out, P::Union{MarslandShardlow, MarslandShardlowAux})
    #zero!(out.q)
    zero!(out)
    out.p .= dm*P.γ
    out
end


"""
    Bridge.a(t,  P::Union{MarslandShardlow, MarslandShardlowAux})

Returns matrix a(t) for Marsland-Shardlow model
"""
function Bridge.a(t,  P::Union{MarslandShardlow, MarslandShardlowAux})
    I = Int[]
    X = UncF[]
    γ2 = P.γ^2
    for i in 1:P.n
            push!(I, 2i)
            push!(X, γ2*one(UncF))
    end
    sparse(I, I, X, 2P.n, 2P.n)
end
Bridge.a(t, x, P::Union{MarslandShardlow, MarslandShardlowAux}) = Bridge.a(t, P)



"""
    σ̃(t,  P::Union{MarslandShardlow, MarslandShardlowAux})

Return sparse matrix  matrix σ̃(t)
"""
function σ̃(t,  P::Union{MarslandShardlow, MarslandShardlowAux})
    Iind = Int[]
    Jind = Int[]
    X = UncF[]
    γ = P.γ
    for i in 1:P.n
        push!(Iind, 2i)
        push!(Jind,i)
        push!(X, γ*one(UncF))
    end
    sparse(Iind, Jind, X, 2P.n, P.n)
end



"""
    amul(t, x::State, xin::State, P::Union{MarslandShardlow, MarslandShardlowAux})

Multiply a(t,x) times xin (which is of type state)
Returns variable of type State
"""
function amul(t, x::State, xin::State, P::Union{MarslandShardlow, MarslandShardlowAux})
    out = copy(xin)
    zero!(out.q)
    out.p .= P.γ^2 .* xin.p
    out
end
function amul(t, x::State, xin::Vector{<:Point}, P::Union{MarslandShardlow, MarslandShardlowAux})
    out = copy(x)
    zero!(out.q)
    out.p .= P.γ^2 .* vecofpoints2state(xin).p
    out
end

########################################################################################################################################################################################
################ AHS model #########################################################################################

"""
    K̄(q,τ)

K̄(q,τ) = exp(-Bridge.inner(q)/(2*τ^2))
Kernel for noisefields of AHS-model
"""
function K̄(q,τ)
     exp(-Bridge.inner(q)/(2*τ^2))
end

"""
    ∇K̄(q,τ)

Gradient of kernel for noisefields
"""
function ∇K̄(q,τ)
     -τ^(-2) * K̄(q,τ) * q
end

"""
    ∇K̄(q, qT, τ)

Needed for b! in case P is auxiliary process
"""
function ∇K̄(q, qT, τ)
    -τ^(-2) * K̄(qT,τ) * q
end

"""
    Define z(q) = < ∇K̄(q - δ,τ), λ >
    Required for Stratonovich -> Ito correction in AHS-model
"""
z(q,τ,δ,λ) =  Bridge.inner(∇K̄(q - δ,τ),λ)

"""
    Define ∇z(q) = ∇ < ∇K̄(q - δ,τ), λ >
    Required for Stratonovich -> Ito correction in AHS-model
"""
∇z(q,τ,δ,λ) =  ForwardDiff.gradient(x -> z(x,τ,δ,λ),q)

# function for specification of diffusivity of landmarks
"""
    σq(q, nf::Noisefield) = Diagonal(nf.γ * K̄(q - nf.δ,nf.τ))

Suppose one noise field nf
Returns diagonal matrix with noisefield for position at point location q (can be vector or Point)
"""
σq(q, nf::Noisefield) = Diagonal(nf.γ * K̄(q - nf.δ,nf.τ))

"""
    σp(q, p, nf::Noisefield) = -Diagonal(p .* nf.γ .* ∇K̄(q - nf.δ,nf.τ))

Suppose one noise field nf
Returns diagonal matrix with noisefield for momentum at point location q (can be vector or Point)
"""
σp(q, p, nf::Noisefield) = -Diagonal(p .* nf.γ .* ∇K̄(q - nf.δ,nf.τ))


"""
    For AHS model compute total noise field on position experienced at a point x.
    Useful for plotting purposes.

    Example usage:
        σq(Point(0.0, 0.0), nfs)
        σq([0.0; 0.0], nfs)
"""
function σq(x, nfs::Array{<:Noisefield,1})
    out = σq(x, nfs[1])
        for j in 2:length(nfs)
            out += σq(x, nfs[j])
    end
    out
end

σq(nfs) = (x) -> σq(x,nfs) # inefficient

"""
    construct_nfs(db::Array{Float64,1}, nfstd, γ)

Construct sequence of Noisefields for AHS model
db: domainbound. Vector [db[1], db[2]] (in case d=2), where sources are places on square grid specified by
        (-db[1]:2nfstd:db[1]) x -db[2]:2nfstd:db[2]
    nfstd: standard deviation of noise fields (the smaller: the more noise fields we use)
    γ: if set to one, then the value of the  noise field on the positions is approximately 1 at all locations in the domain
"""
function construct_nfs(db::Array{Float64,1}, nfstd, γ)
    r1 = -db[1]:2nfstd:db[1]
    if d==1
        nfloc = Point.(collect(r1))[:]
        nfscales = [2/pi*γ*Point(1.0) for x in nfloc]  # intensity
    elseif d==2
        r2 = -db[2]:2nfstd:db[2]
        nfloc = Point.(collect(product(r1, r2)))[:]
        nfscales = [2/pi*γ*Point(1.0, 1.0) for x in nfloc]  # intensity
    elseif d==3
        error("first test carefully whether construct_nfs is ok for d=3")
        r2 = -db[2]:2nfstd:db[2]
        r3 = -db[3]:2nfstd:db[3]
        nfloc = Point.(collect(product(r1, r2,r3)))[:]
        nfscales = [2/pi*γ*Point(1.0, 1.0, 1.0) for x in nfloc]  # intensity
    end
    [Noisefield(δ, λ, nfstd) for (δ, λ) in zip(nfloc, nfscales)]
end


"""
Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::Landmarks)
    zero!(out)
    for i in 1:P.n
        for j in 1:P.n
            out.q[i] += p(x,j)*kernel(q(x,i) - q(x,j), P)
            out.p[i] +=  -dot(p(x,i), p(x,j)) * ∇kernel(q(x,i) - q(x,j), P)
        end
         if itostrat
             for k in 1:length(P.nfs)
                 nf = P.nfs[k]
                 out.q[i] += 0.5 * z(q(x,i),nf.τ,nf.δ,nf.γ) * K̄(q(x,i)-nf.δ,nf.τ) * nf.γ
                 out.p[i] += 0.5 * dot(p(x,i),nf.γ) * ( z(q(x,i),nf.τ,nf.δ,nf.γ) * ∇K̄(q(x,i)-nf.δ,nf.τ) -K̄(q(x,i)-nf.δ,nf.τ) * ∇z(q(x,i),nf.τ,nf.δ,nf.γ) )
             end
         end
    end
    out
end

"""
    Bridge.b!(t, x, out, Paux::LandmarksAux)

Evaluate drift of landmarks auxiliary process in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, Paux::LandmarksAux)
    zero!(out)
    for i in 1:Paux.n
        for j in 1:Paux.n
            out.q[i] += p(x,j)*kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)
        end
        if itostrat
            for k in 1:length(Paux.nfs)
                # approximate q by qT
                nf = Paux.nfs[k]
                qT = q(Paux.xT,i)
                out.q[i] += 0.5 * z(qT,nf.τ,nf.δ,nf.γ) * K̄(qT-nf.δ,nf.τ) * nf.γ
                out.p[i] += 0.5 * dot(p(x,i),nf.γ) * ( z(qT,nf.τ,nf.δ,nf.γ) * ∇K̄(qT - nf.δ,nf.τ) -K̄(qT - nf.δ,nf.τ) * ∇z(qT,nf.τ,nf.δ,nf.γ) )
            end
        end
    end
    out
end


"""
    Bridge.B(t, Paux::LandmarksAux)

Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::LandmarksAux)
    X = zeros(UncF, 2Paux.n, 2Paux.n)
    for i in 1:Paux.n
        for j in 1:Paux.n
            X[2i-1,2j] =  kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux) * one(UncF)
        end
        if itostrat
            for k in 1:length(Paux.nfs)
                nf = Paux.nfs[k]
                qT = q(Paux.xT,i)
                X[2i,2i] += 0.5 * ( z(qT,nf.τ,nf.δ,nf.γ) * ∇K̄(qT - nf.δ,nf.τ) -K̄(qT - nf.δ,nf.τ) * ∇z(qT,nf.τ,nf.δ,nf.γ) )  * nf.γ'
            end
        end
    end
    X
end

"""
    Bridge.B!(t,X,out, Paux::LandmarksAux)

Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::LandmarksAux)
    out .= 0.0 * out
    u = zero(UncF)
    for i in 1:Paux.n  # separately loop over even and odd indices
        for k in 1:2Paux.n # loop over all columns
            for j in 1:Paux.n
                 out[2i-1,k] += kernel(q(Paux.xT,i) - q(Paux.xT,j), Paux)*X[p(j), k]
            end
            if itostrat
                u = 0.0*u
                for k in 1:length(Paux.nfs)
                    nf = P.nfs[k]
                    qT = q(Paux.xT,i)
                    u += 0.5 * ( z(qT,nf.τ,nf.δ,nf.γ) * ∇K̄(qT - nf.δ,nf.τ) -K̄(qT - nf.δ,nf.τ) * ∇z(qT,nf.τ,nf.δ,nf.γ) )  * nf.γ'
                end
                out[2i,k] = u * X[2i,k]
            end
        end
    end
    out
end

function Bridge.β(t, Paux::LandmarksAux)
    out = zeros(PointF,Paux.n)
    if itostrat
        for i in 1:Paux.n
            for k in 1:length(Paux.nfs)
                nf = Paux.nfs[k]
                qT = q(Paux.xT,i)
                out[i] += 0.5 * z(qT,nf.τ,nf.δ,nf.γ) * K̄(qT-nf.δ,nf.τ) * nf.γ # simply take q at endpoint
            end
        end
        return(State(out,zeros(PointF,Paux.n)))
    else
        return (State(zeros(PointF,Paux.n), zeros(PointF,Paux.n)) )
    end
end




"""
    Bridge.σ!(t, x_, dm, out, P::Union{Landmarks,LandmarksAux})

Compute sigma(t,x) * dm where dm is a vector and sigma is the diffusion coefficient of landmarks
write to out which is of type State
"""
function Bridge.σ!(t, x_, dm, out, P::Union{Landmarks,LandmarksAux})
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    zero!(out)
    for i in 1:P.n
        for j in 1:length(P.nfs)
            out.q[i] += σq(q(x, i), P.nfs[j]) * dm[j]
            out.p[i] += σp(q(x, i), p(x, i), P.nfs[j]) * dm[j]
        end
    end
    out
end

"""
Compute sigma(t,x)' * y where y is a state and sigma is the diffusion coefficient of landmarks
returns a vector of points of length P.nfs
"""
function σtmul(t, x_, y::State{Pnt}, P::Union{Landmarks,LandmarksAux}) where Pnt
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    out = zeros(Pnt, length(P.nfs))
    for j in 1:length(P.nfs)
        for i in 1:P.n
            out[j] += σq(q(x, i), P.nfs[j])' * y.q[i] +
                        σp(q(x, i), p(x, i), P.nfs[j])' * y.p[i]
        end
    end
    out
end

"""
    σ̃(t,  Paux::LandmarksAux)

Return matrix σ̃(t) for LandmarksAux
"""
function σ̃(t,  Paux::LandmarksAux)
    x = Paux.xT
    out = zeros(UncF, 2Paux.n, length(Paux.nfs))
    for i in 1:Paux.n
        for j in 1:length(Paux.nfs)
            out[2i-1,j] = σq(q(x, i), Paux.nfs[j])
            out[2i,j] = σp(q(x, i), p(x, i), Paux.nfs[j])
        end
    end
    out
end




"""
    σt!(t, x_, y::State{Pnt}, out, P::MarslandShardlow) where Pnt

compute σ(t,x)' y, where y::State
the result is a vector of points that is written to out
"""
function σt!(t, x_, y::State{Pnt}, out, P::MarslandShardlow) where Pnt
    zero!(out)
    out .= P.γ * y.p
    out
end



"""
    compute σ(t,x)' y, where y::State
    the result is a vector of points that is written to out
"""
function σt!(t, x_, y::State{Pnt}, out, P::Union{Landmarks,LandmarksAux}) where Pnt
    zero!(out)
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    for j in 1:length(P.nfs)
        for i in 1:P.n
            out[j] += σq(q(x, i), P.nfs[j])' * q(y, i) +
                    σp(q(x, i), p(x, i), P.nfs[j])' * p(y, i)
        end
    end
    out
end

function Bridge.a(t, x_, P::Union{Landmarks,LandmarksAux})
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    out = zeros(Unc{deepeltype(x)}, 2P.n,2P.n)
    @inbounds for i in 1:P.n, k in i:P.n,  j in 1:length(P.nfs)
                a11 = σq(q(x,i),P.nfs[j])
                a21 = σp(q(x,i),p(x,i),P.nfs[j])
                a12 = σq(q(x,k),P.nfs[j])
                a22 = σp(q(x,k),p(x,k),P.nfs[j])
                out[2i-1,2k-1] += a11 * a12'
                out[2i-1,2k] += a11 * a22'
                out[2i,2k-1] += a21 * a12'
                out[2i,2k] += a21 * a22'
    end
    @inbounds for i in 2:2P.n,  k in 1:i-1
            out[i,k] = out[k,i]
    end
    out
end

Bridge.a(t, P::LandmarksAux) =  Bridge.a(t, 0, P)


"""
    amul(t, x::State, xin::Vector{<:Point}, P::Union{Landmarks,LandmarksAux})

Multiply a(t,x) times a vector of points
Returns a State
(first multiply with sigma', via function σtmul, next left-multiply this vector with σ)
"""
function amul(t, x::State, xin::Vector{<:Point}, P::Union{Landmarks,LandmarksAux})
    #vecofpoints2state(Bridge.a(t, x, P)*xin)
    out = copy(x)
    zero!(out)
    Bridge.σ!(t, x, σtmul(t, x, vecofpoints2state(xin), P),out,P)
end

"""
Multiply a(t,x) times a state
Returns a state
(first multiply with sigma', via function σtmul, next left-multiply this vector with σ)
"""
function amul(t, x::State, xin::State, P::Union{Landmarks,LandmarksAux})
    #vecofpoints2state(Bridge.a(t, x, P)*vec(xin))
    out = copy(x)
    zero!(out)
    Bridge.σ!(t, x, σtmul(t, x, xin, P),out,P)
end

"""
    Bridge.a!(t, x_, out, P::Union{Landmarks,LandmarksAux})
"""
function Bridge.a!(t, x_, out, P::Union{Landmarks,LandmarksAux})
    zero!(out)
    if P isa Landmarks
        x = x_
    else
        x = P.xT
    end
    @inbounds for i in 1:P.n,  k in i:P.n, j in 1:length(P.nfs)
        a11 = σq(q(x,i),P.nfs[j])
        a21 = σp(q(x,i),p(x,i),P.nfs[j])
        a12 = σq(q(x,k),P.nfs[j])
        a22 = σp(q(x,k),p(x,k),P.nfs[j])
        out[2i-1,2k-1] += a11 * a12'
        out[2i-1,2k] += a11 * a22'
        out[2i,2k-1] += a21 * a12'
        out[2i,2k] += a21 * a22'
    end
    @inbounds for i in 2:2P.n, k in 1:i-1
            out[i,k] = out[k,i]
    end
    out
end


function getγ(P)
    if isa(P,MarslandShardlow)
        out = P.γ
    elseif isa(P,Landmarks)
        out = P.nfs[1].γ[1]*pi/2
    end
    out
end

function dimwiener(P)
    if P isa MarslandShardlow
        out = P.n
    elseif P isa Landmarks
        out = length(P.nfs)
    end
    out
end

"""
    gramkernel(q, P; ϵ = 10^(-12))

Gram matrix for kernel with vector of landmarks given by q::Vector(PointF)
ϵ*I is added to avoid numerical problems that destroy PSDness of the Gram matrix
"""
function gramkernel(q, P; ϵ = 10^(-12))
    K =  [(kernel(q[i]- q[j],P) + (i==j)*ϵ) * one(UncF)   for i  in eachindex(q), j in eachindex(q)]
    PDMat(deepmat(K))
end


function hamiltonian(x::NState, P::MarslandShardlow)
    s = 0.0
    for i in 1:P.n, j in 1:P.n
        s += dot(x.p[i], x.p[j])*kernel(x.q[i] - x.q[j], P)
    end
    0.5 * s
end
