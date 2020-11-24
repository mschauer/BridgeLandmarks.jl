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
struct GuidRecursions{TL,TM⁺,TM, Tμ, TH, TLt0, TMt⁺0, Tμt0}
    Lt::Vector{TL}          # Lt on grid tt
    Mt⁺::Vector{TM⁺}        # Mt⁺ on grid tt
    Mt::Vector{TM}          # Mt on grid tt
    μt::Vector{Tμ}          # μt on grid tt
    Ht::Vector{TH}          # Ht on grid tt
    Lt0::TLt0               # Lt at time 0, after gpupdate step incorporating observation xobs0
    Mt⁺0::TMt⁺0             # inv(Mt) at time 0, after gpupdate step incorporating observation xobs0
    μt0::Tμt0               # μt at time 0, after gpupdate step incorporating observation xobs0

    function GuidRecursions(t,obsinfo)
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
        new{eltype(Lt), eltype(Mt⁺), eltype(Mt),eltype(μt),eltype(Ht), typeof(Lt0), typeof(Mt⁺0), typeof(μt0)}(Lt, Mt⁺,Mt, μt, Ht, Lt0, Mt⁺0, μt0)
    end
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
    nothing
end


struct Lm  end

"""
    guidingbackwards!(::Lm, t, (Lt, Mt⁺, μt), Paux, obsinfo; implicit=true, lowrank=false)

Solve backwards recursions in L, M, μ parametrisation on grid t

## Arguments
- `t`: time grid
- `(Lt, Mt⁺, μt)`: containers to write the solutions into
- `Paux`: auxiliary process
- `obsinfo`: of type ObsInfo containing information on the observations
- `implicit`: if true an implicit Euler backwards scheme is used (else explicit forward)

Case `lowrank=true` still gives an error: fixme!
"""
function guidingbackwards!(::Lm, t, (Lt, Mt⁺, μt), Paux, obsinfo, xT; implicit=true, lowrank=false) #FIXME: add lowrank
    Mt⁺[end] .= obsinfo.ΣT
    Lt[end] .= obsinfo.LT
    μt[end] .= μT = zeros(PointF,obsinfo.n)

    B̃ = Bridge.B(0, Paux, xT)          # does not depend on time
    β̃ = vec(Bridge.β(0,Paux, xT))       # does not depend on time
    σ̃T = Matrix(σ̃(0, Paux, xT))
    dt = t[2] - t[1]
    oldtemp = (0.5*dt) * Bridge.outer(Lt[end] * σ̃T)
    if lowrank  # TBA lowrank on σ̃T, and write into σ̃T
        error("not implemented")
    end
    for i in length(t)-1:-1:1
        dt = t[i+1]-t[i]
        if implicit
            Lt[i] .= Lt[i+1]/lu(I - dt* B̃, Val(false)) # should we use pivoting?
        else
            Lt[i] .=  Lt[i+1] * (I + B̃ * dt)
        end
        temp = (0.5dt) * Bridge.outer( (Lt[i]) * σ̃T)
        Mt⁺[i] .= Mt⁺[i+1] + oldtemp + temp
        oldtemp = temp
        μt[i] .= μt[i+1] + 0.5 * (Lt[i] + Lt[i+1]) * β̃ * dt  # trapezoid rule
    end
    (Lt[1], Mt⁺[1], μt[1])
end

"""
    update_guidrec!(Q, obsinfo)

Q::GuidedProposal
obsinfo::ObsInfo

Computes backwards recursion for (L,M⁺,μ), including gp-update step at time 0.
Next, the `guidrec` field or `Q` is updated.
Returns `Q`.
"""
function update_guidrec!(Q, obsinfo)
    for k in 1:obsinfo.nshapes  # for all shapes
        # solve backward recursions;
        Lt0₊, Mt⁺0₊, μt0₊ =  guidingbackwards!(Lm(), Q.tt, (Q.guidrec[k].Lt, Q.guidrec[k].Mt⁺,Q.guidrec[k].μt), Q.aux[k], obsinfo, auxiliary(Q,k).xT)
        # perform gpupdate step at time zero
        gp_update!(Lt0₊, Mt⁺0₊, μt0₊, (obsinfo.L0, obsinfo.Σ0, obsinfo.xobs0),Q.guidrec[k].Lt0, Q.guidrec[k].Mt⁺0, Q.guidrec[k].μt0)
        # compute Cholesky decomposition of Mt at each time on the grid, need to symmetrize gr.Mt⁺; else AHS  gives numerical roundoff errors when mT \neq 0
        S = map(X -> 0.5*(X+X'), Q.guidrec[k].Mt⁺)
        for i in eachindex(S)
            Q.guidrec[k].Mt[i] = InverseCholesky(lchol(S[i]))
            Q.guidrec[k].Ht[i] .= Q.guidrec[k].Lt[i]' * (Q.guidrec[k].Mt[i] * Q.guidrec[k].Lt[i] )
        end
    end
    Q
end
