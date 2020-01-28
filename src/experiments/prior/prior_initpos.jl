K = reshape([BL.kernel(x0.q[i]- x0.q[j],P) * one(UncF) for i in 1:n for j in 1:n], n, n)
dK = PDMat(BL.deepmat(K))  #chol_dK = cholesky(dK)  # then dK = chol_dk.U' * chol_dk.U
ndistr = MvNormal(dK)
r = rand(ndistr)
rs = BL.deepvec2state(vcat(r,r))
BL.q(rs)


K = reshape([kernel(xobs0.q[i]- xobs0.q[j],P) * one(UncF) for i in 1:n for j in 1:n], n, n)
dK = PDMat(deepmat(K))  #chol_dK = cholesky(dK)  # then dK = chol_dk.U' * chol_dk.U
inv_dK = inv(dK)
ndistr = MvNormal(stepsize*inv_dK)

using LowRankApprox
Paux = [BL.auxiliary(P, State(xobsT[k],mT)) for k in 1:nshapes]
BBB = Bridge.B(0,Paux[1])
BB= BL.deepmat(BBB)
psvd(BB)
pq = pqrfact(BB)

L0 = LT = [(i==j) * one(UncF) for i in 1:2:2n, j in 1:2n]  # pick position indices

@time LT * BBB
@time BL.deepmat(LT) * pq


dK = PDMat(BL.deepmat(BL.gramkernel(xobs0,P)))
inv_dK = inv(dK)
κ = 1000.0
prior_momenta = MvNormal(κ*inv(dK))


T = 1.0
dt = 0.0005
t = 0.0:dt:T; tt_ =  tc(t,T)




n = 5
P = MarslandShardlow(0.5, 0.5, 0.8, 0.0, n)
# simulate forward from x0
x0 = BL.vecofpoints2state(randn(PointF,2n))
W = BL.initSamplePath(tt_,  zeros(PointF, BL.dimwiener(P)))
sample!(W, Bridge.Wiener{Vector{PointF}}())
Xf = BL.initSamplePath(tt_,x0)
BL.solve!(EulerMaruyama!(), Xf, x0, W, P)  #@time solve!(StratonovichHeun!(), X, x0, W, P)

# extract endpoint, slip sign in endpoint and simulate forward
temp = Xf.yy[end]
x0back = BL.flipmomenta(temp)
Xback = deepcopy(Xf)
Wmin = W#Bridge.SamplePath(W.tt, reverse(W.yy))
BL.solve!(EulerMaruyama!(), Xback, x0back, Wmin, P)  #@time solve!(StratonovichHeun!(), X, x0, W, P)Wback, Xback = landmarksforward(tt_, x0, P)

# if we flip momenta back, we're back at starting point
print(x0)
println()
print(BL.flipmomenta(Xback.yy[end]))

# so to reconstruct full path
#- shape 1 is forward on [0,0.5] with starting point (q0,p0). Next, the path is translated to [0.5, 1] as is.
#- shape 2 is forward on [0,0.5] with starting point (q0,-p0). This is transformed to a path on [0,0.5] as follows.
#  say the path is Y.
# a) Set X[0] = flipmomenta(Y[end])
# b) simulate forward with the same Wiener increment
