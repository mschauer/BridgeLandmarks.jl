# @enter landmarkmatching(landmarks0, landmarksT)
# Juno.@enter landmarkmatching(landmarks0, landmarksT)


struct Info{TLT,TΣT,TμT,TL0,TΣ0,y}
     LT::TLT
     ΣT::TΣT
     μT::TμT
     L0::TL0
     Σ0::TΣ0

    function ObsInfo(LT,ΣT,μT,L0,Σ0, y)
         new{typeof(LT),typeof(ΣT),typeof(μT),typeof(L0),typeof(Σ0)}(LT,ΣT,μT,L0,Σ0*y)
    end
end

Info(1,2,3,4,5,10)


if false
    using BridgeLandmarks
    F = [(i==j) * one(UncF) for i in 1:5, j in 1:3]  # pick position indices
    F
    struct Test
        F
    end

    obj = Test(F)
    obj
    show(obj.F)
    show(obj)
    import Base.show
    function show(io::IO, mime::MIME"text/plain",obj::Test)
        print(io,mime, obj.F)
    end

    function show(io::IO, obj::Test)
        show(io,obj.F)
    end
    show(obj)
end



mutable struct MyStruct{tX, tW}
	X::tX #Vector{Bridge.SamplePath}
	W::tW
end

X = rand(2,4)
W = [rand(3)]
A = MyStruct(X,W)

function change!(A)
	A.W = [rand(10)]
end

change!(A)
A



A1 = [[1]]
A2 = copy(A1)
A1 === A2
A1[1] === A2[1]
A2[1]  =[6.0]
print(A1)

A1
ismutable(A1)               # true
A3 = deepcopy(A1)
A3 === A1
A1[1] === A3[1]



nshapes=3
xinit=BL.NState(rand(PointF,5),rand(PointF,5))
t = collect(0:.01:1.0)

X = [BL.initSamplePath(t, xinit) for _ in 1:nshapes]
W = [BL.initSamplePath(t,  zeros(PointF, 2)) for _ in 1:nshapes]
for k in 1:nshapes
	sample!(W[k], BL.Wiener{Vector{PointF}}())
end
x = deepvec(xinit)
∇x = deepcopy(x)


Xk = BL.initSamplePath(t, 0.0*xinit)
X[2] = Xk

# memory allocations, actual state at each iteration is (X,W,Q,x,∇x) (x, ∇x are initial state and its gradient)
Xᵒ = deepcopy(X)
Qᵒ = deepcopy(Q)
Wᵒ = initSamplePath(t,  zeros(StateW, dwiener))



struct Test{S,T}
	a::S
	b::T
	function TestS(v,w)
		new{typeof(v),typeof(w)}(v,w)
	end
end

mutable struct MTest{S,T}
	a::S
	b::T
	function MTestS(v,w)
		new{typeof(v),typeof(w)}(v,w)
	end
end


adjust = function(x::Test,anew)
	TestS(anew,x.b)
end

adjustm = function(x::MTestSanew)
	x.a = anew
	x
end

using BenchmarkTools

x = TestS(rand(4),[3.,4.])
xm = MTestS(rand(4),[3.,4.])

@benchmark adjust(x,rand(10_000))

@benchmark adjustm(xm,rand(10_000))


d = [[:innov, 1.0, 1], [:parameter, 1, 3], [:parameter, 0, 10], [:innov, 1.0, 6],[:rmmala_mom, 0, 4]]

updatescheme = [:innov, :mala_mom]

dfacc_names = push!(updatescheme, :iteration)
dfacc = DataFrame(fill([], length(updatescheme)),updatescheme)
push!(df, [1,0,3])

DataFrames.names!(df, updatescheme)

updatescheme = [:innov, :mala_mom]
accinfo = DataFrame(fill([], length(updatescheme)+1),push!(updatescheme,:iteration))



function adaptpcnstep!(ρ, n, accinfo, nshapes, η; adaptskip = 15, targetaccept=0.5)

    recentmean = mean(accinfo[end-adaptskip+1:end])
        # ind1 =  findall(first.(accinfo).==:innov)[end-adaptskip*nshapes+1:end]
        # recent_mean = mean(last.(accinfo)[ind1])
        if recentmean > targetaccept
            u = sigmoid(invsigmoid(ρ) - η(n))
        else
            u = sigmoid(invsigmoid(ρ) + η(n))
        end
		ρ = u
    nothing
end

sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))
invsigmoid(z::Real) = log(z/(1-z))

adaptskip = 15; targetaccept=0.5
η = n -> min(0.2, 10/n)                          # stepsize cooling function
nshapes = 1
accinfo = rand(Bernoulli(0.2),30)
n = 5

ρ = 0.9
adaptpcnstep!(ρ, n, accinfo, nshapes, η)
ρ


struct MyS{T}
	x::T
	y

	function MyS(ξ)
		y = sum(ξ)
		new{typeof(ξ)}(ξ,y)
	end
end
x = State(rand(PointF,3),zeros(PointF,3))
deepx = deepvec(x)
BL.deepvec2state(deepx)-x # check
x.p
x.q

MyS(1:10)
MyS(rand(6))

x = State(rand(PointF,3),zeros(PointF,3))
deepx = deepvec(x)
BL.deepvec2state(deepx)-x # check
x.p
x.q
BL.deepvec(x.q)
vec(x)


testfun = function(p,q)
	norm(p+q)
end

testfun2(q)  = (p) -> testfun(p,q)
x.x[1,:]
ForwardDiff.gradient(testfun2(x.q), x.p)

using BridgeLandmarks

function split_state(x::BL.NState)
	q =  vec(reinterpret(BL.deepeltype(x), x.x[1,:]))
	p =  vec(reinterpret(BL.deepeltype(x), x.x[2,:]))
	q, p
end

y = State(rand(PointF,3),rand(PointF,3))
q, p = split_state(y)
yy = merge_state(q,p)
yy-y

function merge_state(q,p)
	BL.NState(reinterpret(Point{eltype(q)}, q),reinterpret(Point{eltype(p)}, p))
end
