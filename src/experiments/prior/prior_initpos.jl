K = reshape([BL.kernel(x0.q[i]- x0.q[j],P) * one(UncF) for i in 1:n for j in 1:n], n, n)
dK = PDMat(BL.deepmat(K))  #chol_dK = cholesky(dK)  # then dK = chol_dk.U' * chol_dk.U
ndistr = MvNormal(dK)
r = rand(ndistr)
rs = BL.deepvec2state(vcat(r,r))
BL.q(rs)


K = reshape([kernel(x0.q[i]- x0.q[j],P) * one(UncF) for i in 1:n for j in 1:n], n, n)
dK = PDMat(deepmat(K))  #chol_dK = cholesky(dK)  # then dK = chol_dk.U' * chol_dk.U
inv_dK = inv(dK)
ndistr = MvNormal(stepsize*inv_dK)
