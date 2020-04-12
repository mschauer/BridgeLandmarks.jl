"""
    GuidRecursions{TL,TM⁺,TM, Tμ, TH, TLt0, TMt⁺0, Tμt0}

GuidRecursions defines a struct that contains all info required for computing the guiding term and
likelihood (including ptilde term) for a single shape

## Arguments
Suppose t is the specified (fixed) time grid. Then the elements of the struct are:

- `Lt`:     matrices L
- `Mt⁺``:    matrices M⁺ (inverses of M)
- `M`:      matrices M
- `μ``:      vectors μ
- `Ht`:     matrices H, where H = L' M L
- `Lt0`:    L(0) (so obtained from L(0+) after gpupdate step incorporating observation xobs0)
- `Mt⁺0`:   M⁺(0) (so obtained from M⁺(0+) after gpupdate step incorporating observation xobs0)
- `μt0`:    μ(0) (so obtained μ(0+) after gpupdate step incorporating observation xobs0)
"""
mutable struct GuidRecursions{TL,TM⁺,TM, Tμ, TH, TLt0, TMt⁺0, Tμt0}
    Lt::Vector{TL}          # Lt on grid tt
    Mt⁺::Vector{TM⁺}        # Mt⁺ on grid tt
    Mt::Vector{TM}          # Mt on grid tt
    μt::Vector{Tμ}          # μt on grid tt
    Ht::Vector{TH}          # Ht on grid tt
    Lt0::TLt0               # Lt at time 0, after gpupdate step incorporating observation xobs0
    Mt⁺0::TMt⁺0             # inv(Mt) at time 0, after gpupdate step incorporating observation xobs0
    μt0::Tμt0               # μt at time 0, after gpupdate step incorporating observation xobs0

    function GuidRecursions(Lt, Mt⁺,Mt, μt, Ht, Lt0, Mt⁺0, μt0)
            new{eltype(Lt), eltype(Mt⁺), eltype(Mt),eltype(μt),eltype(Ht), typeof(Lt0), typeof(Mt⁺0), typeof(μt0)}(Lt, Mt⁺,Mt, μt, Ht, Lt0, Mt⁺0, μt0)
    end
end


"""
    init_guidrec(t,obs_info,xobs0)

Initialise (allocate memory) a struct of type GuidRecursions for a single shape
"""
function init_guidrec(t,obsinfo)
    μT = zeros(PointF,obsinfo.n)
    Pnt = eltype(obsinfo.ΣT)
    Lt =  [copy(obsinfo.LT) for _ in t]
    Mt⁺ = [copy(obsinfo.ΣT) for _ in t]
    Mt = map(X -> InverseCholesky(lchol(X)),Mt⁺)
    μt = [copy(μT) for _ in t]
    H = obsinfo.LT' * (obsinfo.ΣT * obsinfo.LT )
    Ht = [copy(H) for _ in t]
    Lt0 = copy([obsinfo.L0; obsinfo.LT])

    m = size(obsinfo.Σ0)[1]
    n = size(obsinfo.ΣT)[2]
    if m==0
        Mt⁺0 = copy(obsinfo.ΣT)
    else
        Mt⁺0 = [copy(obsinfo.Σ0) zeros(Pnt,m,n); zeros(Pnt,n,m) copy(obsinfo.ΣT)]
    end
    μt0 = [0*obsinfo.xobs0; copy(μT)]
    GuidRecursions(Lt, Mt⁺, Mt, μt, Ht, Lt0, Mt⁺0, μt0)
end

"""
    gp_update!(Lt0₊, Mt⁺0₊::Array{Pnt,2}, μt0₊, (L0, Σ0, xobs0), Lt0, Mt⁺0, μt0) where Pnt

Guided proposal update for newly incoming observation at time zero.
Information on new observations at time zero is `(L0, Σ0, xobs0)`
Values just after time zero, `(Lt0₊, Mt⁺0₊, μt0₊)` are updated to time zero, the result being written into `(Lt0, Mt⁺0, μt0)`
"""
function gp_update!(Lt0₊, Mt⁺0₊::Array{Pnt,2}, μt0₊, (L0, Σ0, xobs0), Lt0, Mt⁺0, μt0) where Pnt
    Lt0 .= [L0; Lt0₊]
    m = size(Σ0)[1]
    n = size(Mt⁺0₊)[2]
    if m==0
        Mt⁺0 .= Mt⁺0₊
    else
        Mt⁺0 .= [Σ0 zeros(Pnt,m,n); zeros(Pnt,n,m) Mt⁺0₊]
    end
    μt0 .= [0*xobs0; μt0₊]
end


struct Lm  end

"""
    guidingbackwards!(::Lm, t, (Lt, Mt⁺, μt), Paux, obs_info; implicit=true, lowrank=false)

Solve backwards recursions in L, M, μ parametrisation on grid t

## Arguments
- `t`: time grid
- `(Lt, Mt⁺, μt)`: containers to write the solutions into
- `Paux`: auxiliary process
- `obs_info`: of type ObsInfo containing information on the observations
- `implicit`: if true an implicit Euler backwards scheme is used (else explicit forward)

Case `lowrank=true` still gives an error: fixme!
"""
function guidingbackwards!(::Lm, t, (Lt, Mt⁺, μt), Paux, obsinfo; implicit=true, lowrank=false) #FIXME: add lowrank
    Mt⁺[end] .= obsinfo.ΣT
    Lt[end] .= obsinfo.LT
    μt[end] .= μT = zeros(PointF,obsinfo.n)

    B̃ = Bridge.B(0, Paux)          # does not depend on time
    β̃ = vec(Bridge.β(0,Paux))       # does not depend on time
    σ̃T = Matrix(σ̃(0, Paux))
    dt = t[2] - t[1]
    oldtemp = (0.5*dt) * Bridge.outer(Lt[end] * σ̃T)
    if lowrank
        # TBA lowrank on σ̃T, and write into σ̃T
        error("not implemented")
    end

    for i in length(t)-1:-1:1
        dt = t[i+1]-t[i]
        if implicit
            Lt[i] .= Lt[i+1]/lu(I - dt* B̃, Val(false)) # should we use pivoting?
        else
            Lt[i] .=  Lt[i+1] * (I + B̃ * dt)
        end
        temp = (0.5*dt) * Bridge.outer(Lt[i] * σ̃T)
        Mt⁺[i] .= Mt⁺[i+1] + oldtemp + temp
        oldtemp = temp
        μt[i] .= μt[i+1] + 0.5 * (Lt[i] + Lt[i+1]) * β̃ * dt  # trapezoid rule
    end
    (Lt[1], Mt⁺[1], μt[1])
end


"""
    update_guidrec!(Q, obs_info)

Compute backward ODEs required for guided proposals (for all shapes) and write into field `Q.guidrec`
"""
function update_guidrec!(Q, obsinfo)
    Qgr = copy(Q.guidrec)
    for k in 1:obsinfo.nshapes  # for all shapes
        gr = Qgr[k]
        # solve backward recursions;
        Lt0₊, Mt⁺0₊, μt0₊ =  guidingbackwards!(Lm(), Q.tt, (gr.Lt, gr.Mt⁺,gr.μt), Q.aux[k], obsinfo)
        # perform gpupdate step at time zero
        gp_update!(Lt0₊, Mt⁺0₊, μt0₊, (obsinfo.L0, obsinfo.Σ0, obsinfo.xobs0),gr.Lt0, gr.Mt⁺0, gr.μt0)
        # compute Cholesky decomposition of Mt at each time on the grid, need to symmetrize gr.Mt⁺; else AHS  gives numerical roundoff errors when mT \neq 0
        S = map(X -> 0.5*(X+X'), gr.Mt⁺)
        gr.Mt = map(X -> InverseCholesky(lchol(X)),S)
        # compute Ht at each time on the grid
        for i in 1:length(gr.Ht)
            gr.Ht[i] .= gr.Lt[i]' * (gr.Mt[i] * gr.Lt[i] )
        end
    end
    set_guidrec!(Q::GuidedProposal!, Qgr)
end
