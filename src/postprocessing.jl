"""
        Write bridge iterates to file
        Xsave: contains values of bridges
        tt_: grid on which bridges are simulated
        n: nr of bridges
        nshapes: nr of shapes
        subsamples: indices of iterates that are kept and saved in Xsave
        outdir: output directory
"""
function write_mcmc_iterates(Xsave, tt_, n, nshapes, subsamples, outdir)
    nshapes = length(xobsT)
    iterates = reshape(vcat(Xsave...),2*d*length(tt_)*n*nshapes, length(subsamples)) # each column contains samplepath of an iteration
    # Ordering in each column is as follows:
    # 0) shape
    # 1) time
    # 2) landmark nr
    # 3) for each landmark: q1, q2 p1, p2
    pqtype = repeat(["pos1", "pos2", "mom1", "mom2"], length(tt_)*P.n*nshapes)
    times = repeat(tt_,inner=2d*n*nshapes)
    landmarkid = repeat(1:P.n, inner=2d, outer=length(tt_)*nshapes)
    shapes = repeat(1:nshapes, inner=length(tt_)*2d*P.n)

    out = hcat(times,pqtype,landmarkid,shapes,iterates)
    headline = "time " * "pqtype " * "landmarkid " * "shapes " * prod(map(x -> "iter"*string(x)*" ",subsamples))
    headline = chop(headline,tail=1) * "\n"

    fn = outdir*"iterates.csv"
    f = open(fn,"w")
    write(f, headline)
    writedlm(f,out)
    close(f)
end


"""
    Write info to txt file
"""
function write_info(sampler, ITER, n, tt_,σobs, ρ, δ, perc_acc_pcn, outdir)
    f = open(outdir*"info.txt","w")
    write(f, "Sampler: ", string(sampler), "\n")
    write(f, "Number of iterations: ",string(ITER),"\n")
    write(f, "Number of landmarks: ",string(n),"\n")
    write(f, "Length time grid: ", string(length(tt_)),"\n")
    write(f, "Endpoint: ",string(tt_[end]),"\n")
    write(f, "Noise Sigma: ",string(σobs),"\n")
    write(f, "Initialisation of rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
    write(f, "Initialisation of MALA parameter (delta): ",string(δ),"\n")
    write(f, "skip in evaluation of loglikelihood: ",string(sk),"\n")
    write(f, "Average acceptance percentage pCN update steps: ",string(perc_acc_pcn),"\n\n")
    close(f)
end

function write_acc(accinfo,accpcn,outdir)
    # extract number of distinct update steps in accinfo
    nunique = length(unique(map(x->x[1], accinfo)))
    niterates = length(accpcn)
    accdf = DataFrame(kernel = map(x->x.kernel, accinfo), acc = map(x->x.acc, accinfo), iter = repeat(1:niterates, inner= nunique))
    accpcndf = DataFrame(kernel = fill(Symbol("pCN"),niterates), acc=accpcn, iter=1:niterates)
    append!(accdf, accpcndf)
    CSV.write(outdir*"accdf.csv", accdf; delim=";")
end


"""
    Write observations to file
"""
function write_observations(xobs0, xobsT, n, nshapes, x0,outdir)
    valueT = vcat(map(x->deepvec(x), xobsT)...) # merge all observations at time T in one vector
    posT = repeat(["pos1","pos2"], n*nshapes)
    shT = repeat(1:nshapes, inner=d*n)
    obsTdf = DataFrame(pos=posT,shape=shT, value=valueT,landmark=repeat(1:n,inner=d,outer=nshapes))

    q0 = map(x->vec(x),x0.q)
    p0 = map(x->vec(x),x0.p)
    obs0df = DataFrame(pos1=extractcomp(q0,1), pos2=extractcomp(q0,2), mom1=extractcomp(p0,1) , mom2=extractcomp(p0,2),landmark=1:n)

    CSV.write(outdir*"obs0.csv", obs0df; delim=";")
    CSV.write(outdir*"obsT.csv", obsTdf; delim=";")
end


"""
    Write parameter iterates to file
"""
function write_params(parsave,subsamples,outdir)
    parsdf = DataFrame(a=extractcomp(parsave,1),c=extractcomp(parsave,2),
            gamma=extractcomp(parsave,3), iterate=subsamples)
    CSV.write(outdir*"parameters.csv", parsdf; delim=";")
end

"""
    Write noisefields to file (empty in case of MS-model)
"""
function write_noisefields(P,outdir)
    if isa(P,Landmarks)
        nfsloc = [P.nfs[j].δ for j in eachindex(P.nfs)]
        nfsdf = DataFrame(locx =  extractcomp(nfsloc,1),
                          locy =  extractcomp(nfsloc,2),
                          nfstd=fill(P.nfstd,length(P.nfs)))
    elseif isa(P,MarslandShardlow)
        nfsdf =DataFrame(locx=Int64[], locy=Int64[], nfstd=Int64[])
    end
    CSV.write(outdir*"noisefields.csv", nfsdf; delim=";")
end
