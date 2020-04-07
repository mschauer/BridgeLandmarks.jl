using StaticArrays

function deepmat(H::AbstractMatrix{S}) where {S}
    d1, d2 = size(S)
    reshape([H[i,j][k,l] for k in 1:d1, i in 1:size(H, 1), l in 1:d2, j in 1:size(H,2)], d1*size(H,1), d2*size(H,2))
end
#@test  outer(deepvec(x0)) == deepmat(outer(vec(x0)))


function deepmat2unc(A::Matrix)  # d is the dimension of the square subblocks
  m = div(size(A,1),d)
  n = div(size(A,2),d)
  [Unc(A[(i-1)*d+1:i*d,(j-1)*d+1:j*d]) for i in 1:m, j in 1:n]
end

"""
Good display of variable of type State
"""
function Base.show(io::IO, state::State)
  show(io,  "text/plain", hcat(q(state),p(state)))
end

"""
Solve L L'y =x using two backsolves,
L should be lower triangular
"""
function cholinverse!(L, x)
    LinearAlgebra.naivesub!(L, x) # triangular backsolves
    LinearAlgebra.naivesub!(UpperTriangular(L'), x)
    x
end

lchol(A) = LowerTriangular(Matrix(LinearAlgebra._chol!(copy(A), UpperTriangular)[1])')

############################ InverseCholesky ########################
struct InverseCholesky{T}
    L::T
end

"""
Compute y = H*x where Hinv = L*L' (Cholesky decomposition

Input are L and x, output is y

y=Hx is equivalent to LL'y=x, which can be solved
by first backsolving Lz=x for z and next
backsolving L'y=z

L is a lower triangular matrix with element of type UncMat
x is a State or vector of points
Returns a State (Todo: split up again and return vector for vector input)
"""
function Base.:*(H::InverseCholesky, x::State)
    y = copy(vec(x))
    cholinverse!(H.L,  y) # triangular backsolve
    vecofpoints2state(y)
end

function Base.:*(H::InverseCholesky, x::Vector{<:Point})
    y = copy(x)
    cholinverse!(H.L,  y) # triangular backsolve
end

"""
Compute y = H*X where Hinv = L*L' (Cholesky decomposition

Input are L and X, output is Y

y=HX is equivalent to LL'y=X, which can be solved
by first backsolving LZ=X for z and next
backsolving L'Y=Z

L is a lower triangular matrix with element of type UncMat
X is a matrix with elements of type UncMat
Returns a matrix with elements of type UncMat
"""
function Base.:*(H::InverseCholesky, X)
    cholinverse!(H.L,  copy(X)) # triangular backsolves
end

# convert dual to float, while retaining float if type is float
deepvalue(x::Float64) = x
deepvalue(x::ForwardDiff.Dual) = ForwardDiff.value(x)

"""
    deepvalue(x)

If `x` is a vector of Float64, `x` is returned. If `x` is a vector of Dual-numbers, its Float64 part is returned.
"""
deepvalue(x) = deepvalue.(x)

"""
    deepvalue(x::State)

Extract Float64 part of elements in State (so in case of Dual numbers, derivative part is dropped)
"""
function deepvalue(x::State)
    State(deepvalue.(x.x))
end
