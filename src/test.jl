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


struct TestS2{T}
	a::T
	b::T
	function TestS(v,w,s)
		new{typeof(v)}(sin.(v+w),cos.(s))
	end
end

TestS2([1.,2.],[3.,4.],[5.,6.])
