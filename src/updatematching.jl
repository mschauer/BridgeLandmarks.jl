# routines to determine correct matching of landmarks (cyclic shift detection)

"""
    update_cyclicmatching(X, ll,obs_info, Xᵒ, W, Q)

## writes into / modifies
- `X`
- `ll`

## Returns
`Q`, `obsinfo`, `accept`
"""
function update_cyclicmatching(X, ll,obsinfo, Xᵒ, W, Q)
    direction = rand(Bool)

    # shift each element in xobsT, this means
        # - creating xobsTᵒ (shift each element of xobsT)
        # - creating obsinfoᵒ
        # - creating  Qᵒ from Q by using xobsTᵒ and obsinfoᵒ
        # - recomputing backward recursion

    direction = 2 * rand(Bool) -1
    #    xobsTᵒ = Base.circshift.(Q.xobsT,direction) # this would be for shifting all shapes
    xobsTᵒ = copy(Q.xobsT)
    k = sample(1:Q.nshapes) # randomly pick a shape to which we cyclicallyl shift indices
    xobsTᵒ[k] = Base.circshift(Q.xobsT[k],direction)
    oi = obsinfo
    obsinfoᵒ = ObsInfo(oi.L0, oi.LT, oi.Σ0, oi.ΣT, oi.xobs0, xobsTᵒ, oi.obs_atzero, oi.fixinitmomentato0, oi.n, oi.nshapes)
    Qᵒ = construct_gp_xobsT(Q, xobsTᵒ)
    update_guidrec!(Qᵒ, obsinfoᵒ)

    x0 = X[1].yy[1]
    llᵒ = gp!(LeftRule(), Xᵒ, x0, W, Qᵒ; skip=sk)
    if log(rand()) <= (sum(llᵒ) - sum(ll))
        ll .= llᵒ
        Q = Qᵒ
        obsinfo = obsinfoᵒ
        for k in 1:Q.nshapes
            for i in eachindex(X[1].yy)
                X[k].yy[i] .= Xᵒ[k].yy[i]
            end
        end
        accept = 1
        println("Cyclic shift for shape $k in direction $direction.")
    else
        accept = 0
    end
    Q, obsinfo, accept
end
