"""
    struct containing information on the observations

We assue observations V0 and VT, where
- V0 = L0 * X0 + N(0,Σ0)
- VT = LT * X0 + N(0,ΣT)
In addition, μT is a vector of zeros (for initialising the backward ODE on μ) (possibly remove later)
"""
struct ObsInfo{TLT,TΣT,TL0,TΣ0,Txobs0,TxobsT}
     L0::TL0
     LT::TLT
     Σ0::TΣ0
     ΣT::TΣT
     xobs0::Txobs0
     xobsT::TxobsT
     obs_atzero::Bool
     fixinitmomentato0::Bool
     n::Int64
     nshapes::Int64
end


"""
    set_obsinfo(xobs0,xobsT,Σobs, obs_atzero::Bool,fixinitmomentato0::Bool)

Constructor for ObsInfo.

## Arguments
- `n`: number of landmarks
- `obs_atzero`: Boolean, if true, the initial configuration is observed
- `fixinitmomenta0`: Boolean, if true, the initial momenta are fixed to zero
- `Σobs`: 2-element array where Σobs0 = Σobs[1] and ΣobsT = Σobs[2]
    Both Σobs0 and ΣobsT are arrays of length n of type UncF that give the observation covariance matrix on each landmark
- `xobs0`: in case obs_atzero=true, it is provided and passed through; in other cases it is constructed such that the backward ODEs are initialised correctly.

Note that there are three cases:
- `obs_atzero=true`: this refers to the case of observing one landmark configuration at times 0 and T
- `obs_atzero=false & fixinitmomentato0=false`: case of observing multiple shapes at time T,
    both positions and momenta at time zero assumed unknown
- `obs_atzero=false & fixinitmomentato0=true`: case of observing multiple shapes at time T,
    positions at time zero assumed unknown, momenta at time 0 are fixed to zero
"""
function set_obsinfo(xobs0,xobsT,Σobs, obs_atzero::Bool,fixinitmomentato0::Bool)
    Σobs0 = Σobs[1]; ΣobsT = Σobs[2]
    n = length(xobsT[1])
    nshapes = length(xobsT)
    if obs_atzero # don't update initial positions, but update initialmomenta
        L0 = LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n]  # pick position indices
        Σ0 = [(i==j) * Σobs0[i] for i in 1:n, j in 1:n]
        ΣT = [(i==j) * ΣobsT[i] for i in 1:n, j in 1:n]
    elseif !obs_atzero & !fixinitmomentato0  # update initial positions and initial momenta
        L0 = Array{UncF}(undef,0,2*n)
        Σ0 = Array{UncF}(undef,0,0)
        xobs0 = Array{PointF}(undef,0)
        LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n]
        ΣT = [(i==j) * ΣobsT[i] for i in 1:n, j in 1:n]
    elseif !obs_atzero & fixinitmomentato0   # only update positions and fix initial state momenta to zero
        xobs0 = zeros(PointF,n)
        L0 = [((i+1)==j) * one(UncF) for i in 1:2:2n, j in 1:2n] # pick momenta indices
        LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n] # pick position indices
        Σ0 = [(i==j) * Σobs0[i] for i in 1:n, j in 1:n]
        ΣT = [(i==j) * ΣobsT[i] for i in 1:n, j in 1:n]
    end
    #μT = zeros(PointF,n)
    ObsInfo(L0,LT,Σ0,ΣT,xobs0,xobsT,obs_atzero,fixinitmomentato0,n,nshapes)
end
