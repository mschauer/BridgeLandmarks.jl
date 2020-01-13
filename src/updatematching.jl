# routines to determine correct matching of landmarks (cyclic shift detection)

### update_matching

function update_matching(obs_info,X, Xᵒ,W, Q, Qᵒ, x, ll)
    direction = rand(Bool)
    Qᵒ.xobsT = circshift.(Q.xobsT,direction)
    update_guidrec!(Qᵒ, obs_info)   # compute backwards recursion
    llᵒ = simguidedlm_llikelihood!(LeftRule(), Xᵒ, deepvec2state(x), W, Qᵒ; skip=sk)

    A = sum(llᵒ) - sum(ll)

    if log(rand()) <= A
        ll .= llᵒ
        deepcopyto!(Q.guidrec,Qᵒ.guidrec)
        #Q.xobsT = Qᵒ.xobsT
        deepcopyto!(Q.xobsT,Qᵒ.xobsT)
        deepcopyto!(X,Xᵒ)
        accept = 1
    else
        accept = 0
    end
    direction, (kernel = "matchingupdate", acc = accept)
end



#import Base: circshift
#circshift(x::NState) = NState(circshift(x.x,(0,1)))
"""
    Circular shift.
    Move right if direction==true, else move left.
"""
function circshift(x::Vector,direction::Bool)
    intdir = 2 * Int(direction) - 1
     x[circshift(1:length(x),intdir)]
end
