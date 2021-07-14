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
-  `gramT`: Gram-matrix at xT
"""
struct MarslandShardlowAux{S,T} <: ContinuousTimeProcess{State{PointF}}
    a::T        # kernel std parameter
    c::T        # kernel multiplicate parameter
    γ::T        # noise level
    λ::T        # mean reversion
    xT::State{Point{S}}  # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int
    gramT::Matrix{T} # Gram-matrix at xT
end

struct Noisefield{T}
    δ::Point{T}         # locations of noise field
    γ::Point{T}         # scaling at noise field (used to be called lambda)
    τ::T                # std of Gaussian kernel noise field
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
    a::T            # kernel std
    c::T            # kernel multiplicate parameter
    n::Int64        # numer of landmarks
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
    a::T            # kernel std
    c::T            # kernel multiplicate parameter
    xT::State{Point{S}}     # use x = State(P.v, zero(P.v)) where v is conditioning vector
    n::Int64        # numer of landmarks
    nfs::Vector{Noisefield{S}}  # vector containing pars of noisefields
    gramT::Matrix{T} # Gram-matrix at xT
end

"""
    MarslandShardlowAux(P::MarslandShardlow, xT, gramT) = MarslandShardlowAux(P.a,P.c, P.γ, P.λ, xT, P.n, gramT)
"""
MarslandShardlowAux(P::MarslandShardlow, xT, gramT) = MarslandShardlowAux(P.a,P.c, P.γ, P.λ, xT, P.n, gramT)

"""
    LandmarksAux(P::Landmarks, xT, gramT) = LandmarksAux(P.a,P.c, xT, P.n, P.nfs, gramT)
"""
LandmarksAux(P::Landmarks, xT, gramT) = LandmarksAux(P.a,P.c, xT, P.n, P.nfs, gramT)

"""
    auxiliary(P::Union{MarslandShardlow, Landmarks}, xT, gramT)

Construct auxiliary process corresponding to `P` and `xT`
"""
function auxiliary(P::Union{MarslandShardlow, Landmarks}, xT, gramT)
    if isa(P,MarslandShardlow)
        return MarslandShardlowAux(P, xT, gramT)
    elseif isa(P,Landmarks)
        return LandmarksAux(P, xT, gramT)
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
        s += 1/2*dot(x.p[i], x.p[j])*kernel(x.q[i] - x.q[j], P)
    end
    s
end

Bridge.b(t::Float64, x, P::Union{Landmarks,MarslandShardlow})= Bridge.b!(t, x, copy(x), P)

Bridge.σ(t, x, dm, P) =  Bridge.σ!(t, x, dm , 0*x, P)


################ MS model #########################################################################################

"""
    Bridge.b!(t, x, out, P::MarslandShardlow)

Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::MarslandShardlow)
    zero!(out)
    pp = out.p
    qq = out.q
    @inbounds  for i in 1:P.n
        for j in 1:P.n
            qq[i] += p(x,j) * kernel(q(x,i) - q(x,j), P)
            pp[i] += -P.λ * qq[i] - dot(p(x,i), p(x,j)) * ∇kernel(q(x,i) - q(x,j), P)
        end
    end
    out
end

"""
    Bridge.b!(t, x, out, Paux::MarslandShardlowAux)

Evaluate drift of landmarks auxiliary process in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, Paux::MarslandShardlowAux, xT)
    zero!(out)
    pp = out.p
    qq = out.q
    @inbounds  for i in 1:Paux.n
        for j in 1:Paux.n
            qq[i] += p(x,j) * Paux.gramT[i,j]
        end
    end
    pp .= - Paux.λ * qq
    out
end

"""
    Bridge.B(t, Paux::MarslandShardlowAux)

Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::MarslandShardlowAux, xT) # not AD safe
    X = zeros(UncF, 2Paux.n, 2Paux.n)
    @inbounds for i in 1:Paux.n
        for j in 1:Paux.n
            X[2i-1,2j] =  Paux.gramT[i,j] * one(UncF)
            X[2i,2j] = -Paux.λ * Paux.gramT[i,j] * one(UncF)
        end
    end
    X
end

"""
    Bridge.B!(t,X,out, Paux::MarslandShardlowAux)

Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t,X,out, Paux::MarslandShardlowAux, xT)
    out .= 0.0 * out
    @inbounds  for i in 1:Paux.n  # separately loop over even and odd indices
        for k in 1:2Paux.n # loop over all columns
            for j in 1:Paux.n
                 out[2i-1,k] += Paux.gramT[i,j] * X[p(j), k]
                 out[2i,k] += -Paux.λ * Paux.gramT[i,j] * X[p(j), k]
            end
        end
    end
    out
end

function Bridge.β(t, Paux::MarslandShardlowAux, xT) # Not AD save
    State(zeros(PointF,Paux.n), zeros(PointF,Paux.n))
end

"""
    Bridge.σ!(t, x, dm, out, P::MarslandShardlow)

Compute σ(t,x) * dm and write to out
"""
function Bridge.σ!(t, x, dm, out, P::MarslandShardlow)
    zero!(out)
    out.p .= dm*P.γ
    out
end

"""
    Bridge.σ!(t, x, dm, out, P::MarslandShardlowAux, xT)

Compute σ(t,x) * dm and write to out
"""
function Bridge.σ!(t, x, dm, out, P::MarslandShardlowAux, xT)
    zero!(out)
    out.p .= dm*P.γ
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
    Bridge.a(t,  P::Union{MarslandShardlow, MarslandShardlowAux})

Returns matrix a(t) for Marsland-Shardlow model
"""
function Bridge.a(t,  P::MarslandShardlow)
    I = Int[]
    X = UncF[]
    γ2 = P.γ^2
    for i in 1:P.n
        push!(I, 2i)
        push!(X, γ2*one(UncF))
    end
    sparse(I, I, X, 2P.n, 2P.n)
end

function Bridge.a(t,  P::MarslandShardlowAux, xT)
    I = Int[]
    X = UncF[]
    γ2 = P.γ^2
    for i in 1:P.n
        push!(I, 2i)
        push!(X, γ2*one(UncF))
    end
    sparse(I, I, X, 2P.n, 2P.n)
end


Bridge.a(t, x, P::MarslandShardlow) = Bridge.a(t, P)
Bridge.a(t, x, P::MarslandShardlowAux, xT) = Bridge.a(t, P, xT)



function Bridge.a!(t, x, A, P::MarslandShardlowAux, xT)
    zero!(A)
    G = γ2*one(UncF)
    for i ∈ (P.n+1):2P.n
        A[i,i] = G
    end
end



"""
    σ̃(t,  P::Union{MarslandShardlow, MarslandShardlowAux})

Return sparse matrix  matrix σ̃(t)
"""
function σ̃(t,  P::MarslandShardlow)
    Iind = Int[]
    Jind = Int[]
    X = UncF[]
    γ = P.γ
    @inbounds for i in 1:P.n
        push!(Iind, 2i)
        push!(Jind,i)
        push!(X, γ*one(UncF))
    end
    sparse(Iind, Jind, X, 2P.n, P.n)
end

function σ̃(t,  P::MarslandShardlowAux, xT)
    Iind = Int[]
    Jind = Int[]
    X = UncF[]
    γ = P.γ
    @inbounds for i in 1:P.n
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
function amul(t, x::State, xin::State, P::MarslandShardlow)
    out = copy(xin)
    zero!(out.q)
    out.p .= P.γ^2 .* xin.p
    out
end
function amul(t, x::State, xin::State, P::MarslandShardlowAux, xT)
    out = copy(xin)
    zero!(out.q)
    out.p .= P.γ^2 .* xin.p
    out
end



function amul(t, x::State, xin::Vector{<:Point}, P::MarslandShardlow)
    out = copy(x)
    zero!(out.q)
    out.p .= P.γ^2 .* vecofpoints2state(xin).p
    out
end

function amul(t, x::State, xin::Vector{<:Point}, P::MarslandShardlowAux, xT)
    out = copy(x)
    zero!(out.q)
    out.p .= P.γ^2 .* vecofpoints2state(xin).p
    out
end



################ AHS model #########################################################################################

"""
    K̄(q,τ)

K̄(q,τ) = exp(-Bridge.inner(q)/(2*τ^2))
Kernel for noisefields of AHS-model
"""
function K̄(q::T,τ) where T
     exp(-Bridge.inner(q)/(2.0*τ*τ))
end

"""
    ∇K̄(q,τ)

Gradient of kernel for noisefields
"""
function ∇K̄(q::T,τ) where T
     -1.0/(τ*τ) * K̄(q,τ) * q
end

"""
    ∇K̄(q, qT, τ)

Needed for b! in case P is auxiliary process
"""
function ∇K̄(q, qT, τ)
    -1.0/(τ*τ) * K̄(qT,τ) * q
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
∇z(q::T,τ,δ,λ) where T =  ForwardDiff.gradient(x -> z(x,τ,δ,λ),q)::T

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
    @inbounds for j in 2:length(nfs)
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
    #return Noisefield{Float64}[Noisefield(δ, λ, nfstd) for (δ, λ) in zip(nfloc, nfscales)]
    [Noisefield(δ, λ, nfstd) for (δ, λ) in zip(nfloc, nfscales)]
end


"""
Evaluate drift of landmarks in (t,x) and save to out
x is a state and out as well
"""
function Bridge.b!(t, x, out, P::Landmarks)
    zero!(out)
    pp = out.p
    qq = out.q
    @inbounds for i in 1:P.n
        for j in 1:P.n
            qq[i] += p(x,j)*kernel(q(x,i) - q(x,j), P)
            pp[i] +=  -dot(p(x,i), p(x,j)) * ∇kernel(q(x,i) - q(x,j), P)
        end
         if itostrat
             for k in 1:length(P.nfs)
                 nf = P.nfs[k]
                 K̄qi = K̄(q(x,i)-nf.δ,nf.τ)
                 zqi = z(q(x,i),nf.τ,nf.δ,nf.γ)
                 qq[i] += (0.5 * zqi * K̄qi) * nf.γ
                 pp[i] += 0.5 * dot(p(x,i),nf.γ) * ( zqi * ∇K̄(q(x,i)-nf.δ,nf.τ) -K̄qi * ∇z(q(x,i),nf.τ,nf.δ,nf.γ) )
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
function Bridge.b!(t, x, out, Paux::LandmarksAux, xT)
    zero!(out)
    pp = out.p
    qq = out.q
    @inbounds for i in 1:Paux.n
        for j in 1:Paux.n
            qq[i] += p(x,j) * Paux.gramT[i,j]
        end
        if itostrat
            qT = q(xT,i)
            p_i = p(x,i)
            for k in 1:length(Paux.nfs)
                # approximate q by qT
                nf = Paux.nfs[k]
                qq[i] += 0.5 * z(qT,nf.τ,nf.δ,nf.γ) * K̄(qT-nf.δ,nf.τ) * nf.γ
                pp[i] += 0.5 * dot(p_i,nf.γ) * ( z(qT,nf.τ,nf.δ,nf.γ) * ∇K̄(qT - nf.δ,nf.τ) -K̄(qT - nf.δ,nf.τ) * ∇z(qT,nf.τ,nf.δ,nf.γ) )
            end
        end
    end
    out
end

"""
    Bridge.B(t, Paux::LandmarksAux, xT)

Compute tildeB(t) for landmarks auxiliary process
"""
function Bridge.B(t, Paux::LandmarksAux, xT)
    X = zeros(UncF, 2Paux.n, 2Paux.n)
    @inbounds  for i in 1:Paux.n
        for j in 1:Paux.n
            X[2i-1,2j] =  Paux.gramT[i,j] * one(UncF)
        end
        if itostrat
            @inbounds for k in 1:length(Paux.nfs)
                nf = Paux.nfs[k]
                qT = q(xT,i)
                X[2i,2i] += 0.5 * ( z(qT,nf.τ,nf.δ,nf.γ) * ∇K̄(qT - nf.δ,nf.τ) -K̄(qT - nf.δ,nf.τ) * ∇z(qT,nf.τ,nf.δ,nf.γ) )  * nf.γ'
            end
        end
    end
    X
end

"""
    Bridge.B!(t,X,out, Paux::LandmarksAux, xT)

Compute B̃(t) * X (B̃ from auxiliary process) and write to out
Both B̃(t) and X are of type UncMat
"""
function Bridge.B!(t, X, out, Paux::LandmarksAux, xT)
    out .= 0.0 * out
    u = zero(UncF)
    @inbounds  for i in 1:Paux.n  # separately loop over even and odd indices
        for k in 1:2Paux.n # loop over all columns
            for j in 1:Paux.n
                 out[2i-1,k] += Paux.gramT[i,j]*X[p(j), k]
            end
            if itostrat
                u = 0.0*u
                for k in 1:length(Paux.nfs)
                    nf = P.nfs[k]
                    qT = q(xT,i)
                    u += 0.5 * ( z(qT,nf.τ,nf.δ,nf.γ) * ∇K̄(qT - nf.δ,nf.τ) -K̄(qT - nf.δ,nf.τ) * ∇z(qT,nf.τ,nf.δ,nf.γ) )  * nf.γ'
                end
                out[2i,k] = u * X[2i,k]
            end
        end
    end
    out
end

function Bridge.β(t, Paux::LandmarksAux, xT)
    out = zeros(PointF,Paux.n)
    if itostrat
        @inbounds  for i in 1:Paux.n
            for k in 1:length(Paux.nfs)
                nf = Paux.nfs[k]
                qT = q(xT,i)
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
function Bridge.σ!(t, x_, dm, out, P::Landmarks)
    x = x_
    zero!(out)
    qq = out.q
    pp = out.p
    @inbounds for i in 1:P.n
        for j in 1:length(P.nfs)
            qq[i] += σq(q(x, i), P.nfs[j]) * dm[j]
            pp[i] += σp(q(x, i), p(x, i), P.nfs[j]) * dm[j]
        end
    end
    out
end

function Bridge.σ!(t, x_, dm, out, P::LandmarksAux, xT)
    x = xT
    zero!(out)
    qq = out.q
    pp = out.p
    @inbounds for i in 1:P.n
        for j in 1:length(P.nfs)
            qq[i] += σq(q(x, i), P.nfs[j]) * dm[j]
            pp[i] += σp(q(x, i), p(x, i), P.nfs[j]) * dm[j]
        end
    end
    out
end



"""
Compute sigma(t,x)' * y where y is a state and sigma is the diffusion coefficient of landmarks
returns a vector of points of length P.nfs
"""
function σtmul(t, x_, y::State{Pnt}, P::Landmarks) where Pnt
    x = x_
    out = zeros(Pnt, length(P.nfs))
    @inbounds for j in 1:length(P.nfs)
        for i in 1:P.n
            out[j] += σq(q(x, i), P.nfs[j])' * y.q[i] +
                        σp(q(x, i), p(x, i), P.nfs[j])' * y.p[i]
        end
    end
    out
end

function σtmul(t, x_, y::State{Pnt}, P::LandmarksAux, xT) where Pnt
    x = xT
    out = zeros(Pnt, length(P.nfs))
    @inbounds for j in 1:length(P.nfs)
        for i in 1:P.n
            out[j] += σq(q(x, i), P.nfs[j])' * y.q[i] +
                        σp(q(x, i), p(x, i), P.nfs[j])' * y.p[i]
        end
    end
    out
end



"""
    σ̃(t,  Paux::LandmarksAux, xT)

Return matrix σ̃(t) for LandmarksAux
"""
function σ̃(t,  Paux::LandmarksAux, xT)
    x = xT
    out = zeros(UncF, 2Paux.n, length(Paux.nfs))
    @inbounds  for i in 1:Paux.n
        for j in 1:length(Paux.nfs)
            out[2i-1,j] = σq(q(x, i), Paux.nfs[j])
            out[2i,j] = σp(q(x, i), p(x, i), Paux.nfs[j])
        end
    end
    out
end




"""
    compute σ(t,x)' y, where y::State
    the result is a vector of points that is written to out
"""
function σt!(t, x_, y::State{Pnt}, out, P::Landmarks) where Pnt
    zero!(out)
    x = x_
    @inbounds for j in 1:length(P.nfs)
         for i in 1:P.n
            out[j] += σq(q(x, i), P.nfs[j])' * q(y, i) +
                    σp(q(x, i), p(x, i), P.nfs[j])' * p(y, i)


        end
    end
    out
end

function σt!(t, x_, y::State{Pnt}, out, P::LandmarksAux, xT) where Pnt
    zero!(out)
    x = xT
    @inbounds for j in 1:length(P.nfs)
         for i in 1:P.n
            out[j] += σq(q(x, i), P.nfs[j])' * q(y, i) +
                    σp(q(x, i), p(x, i), P.nfs[j])' * p(y, i)


        end
    end
    out
end



function Bridge.a(t, x_, P::Landmarks)
    x = x_
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

function Bridge.a(t, x_, P::LandmarksAux, xT)
    x = xT
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



Bridge.a(t, P::LandmarksAux, xT) =  Bridge.a(t, 0, P, xT)

"""
    amul(t, x::State, xin::Vector{<:Point}, P::Union{Landmarks,LandmarksAux})

Multiply a(t,x) times a vector of points
Returns a State
(first multiply with sigma', via function σtmul, next left-multiply this vector with σ)
"""
function amul(t, x::State, xin::Vector{<:Point}, P::Landmarks)
    #vecofpoints2state(Bridge.a(t, x, P)*xin)
    out = copy(x)
    zero!(out)
    Bridge.σ!(t, x, σtmul(t, x, vecofpoints2state(xin), P),out,P)
end

function amul(t, x::State, xin::Vector{<:Point}, P::LandmarksAux, xT)
    #vecofpoints2state(Bridge.a(t, x, P)*xin)
    out = copy(x)
    zero!(out)
    Bridge.σ!(t, x, σtmul(t, x, vecofpoints2state(xin), P),out,P, xT)
end


"""
Multiply a(t,x) times a state
Returns a state
(first multiply with sigma', via function σtmul, next left-multiply this vector with σ)
"""
function amul(t, x::State, xin::State, P::Landmarks)
    out = copy(x)
    zero!(out)
    Bridge.σ!(t, x, σtmul(t, x, xin, P),out,P)
end
function amul(t, x::State, xin::State, P::LandmarksAux, xT)
    out = copy(x)
    zero!(out)
    Bridge.σ!(t, x, σtmul(t, x, xin, P),out,P, xT)
end



"""
    Bridge.a!(t, x_, out, P::Union{Landmarks,LandmarksAux})
"""
function Bridge.a!(t, x_, out, P::Landmarks)
    zero!(out)
    x = x_
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

function Bridge.a!(t, x_, out, P::LandmarksAux, xT)
    zero!(out)
    x = xT
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

"""
    gram_matrix!(out, q, P)

Write into out the Gram matrix for kernel with vector of landmarks given by q::Vector(PointF)
"""
function gram_matrix!(out, q, P)
    for j ∈ 1:P.n, i ∈ j:P.n
        out[i,j] = out[j,i] = kernel(q[i]- q[j],P)
    end
    out
end


"""
    gram_matrix(q, P)

Gram matrix for kernel with vector of landmarks given by q::Vector(PointF)
"""
function gram_matrix(q, P)
    out = zeros(P.n, P.n)
    for j ∈ 1:P.n, i ∈ j:P.n
        out[i,j] = out[j,i] = kernel(q[i]- q[j],P)
    end
    out
end

"""
    hamiltonian(x::NState, P::MarslandShardlow)

Returns Hamiltonian at `x` for deterministic system (no Wiener noise)
"""
function hamiltonian(x::NState, P::MarslandShardlow)
    s = 0.0
    for i in 1:P.n, j in 1:P.n
        s += dot(x.p[i], x.p[j])*kernel(x.q[i] - x.q[j], P)
    end
    0.5 * s
end
