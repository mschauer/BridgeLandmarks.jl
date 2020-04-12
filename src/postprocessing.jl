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
    outdir[end] == "/" && error("provide pathname without trailing '/'")
    iterates = reshape(vcat(Xsave...),2*d*length(tt_)*n*nshapes, length(subsamples)) # each column contains samplepath of an iteration

    # total number of rows is nshapes * length(tt_) * n * (2d)

    # Ordering in each column is as follows:
    # 0) shape
    # 1) time
    # 2) landmark nr
    # 3) for each landmark: q1, q2 p1, p2
    #if !(d==2) error("pqtype in write_mcmc_iterates only implemented in case d=2") end
    shapes = repeat(1:nshapes, inner=length(tt_)*2d*n)
        #times = repeat(tt_,inner=2d*n*nshapes)  # original buggy version when nshapes=1
    times = repeat(tt_, inner=2*d*n, outer=nshapes)
    landmarkid = repeat(1:n, inner=2d, outer=length(tt_)*nshapes)
    if d==1
        pqtype = repeat(["pos1", "mom1"], outer=length(tt_)*n*nshapes)
    elseif d==2
        pqtype = repeat(["pos1", "pos2", "mom1", "mom2"], outer=length(tt_)*n*nshapes)
    elseif d==3
        pqtype = repeat(["pos1", "pos2", "pos3", "mom1", "mom2", "mom3"], outer=length(tt_)*n*nshapes)
    end
    out = hcat(times,pqtype,landmarkid,shapes,iterates)
    headline = "time " * "pqtype " * "landmarkid " * "shapes " * prod(map(x -> "iter"*string(x)*" ",subsamples))
    headline = chop(headline,tail=1) * "\n"

    fn = joinpath(outdir, "iterates.csv")
    f = open(fn,"w")
    write(f, headline)
    writedlm(f,out)
    close(f)
end


"""
    Write info to txt file
"""
function write_info(model,ITER, n, tt_, updatescheme, Σobs, tp, ρ, δ, perc_acc_pcn, elapsed, outdir)
    outdir[end] == "/" && error("provide pathname without trailing '/'")

    f = open(joinpath(outdir,"info.txt"),"w")
    write(f, "Model: ", string(model),"\n")
    write(f, "Number of iterations: ",string(ITER),"\n")
    write(f, "Number of landmarks: ",string(n),"\n")
    write(f, "Length time grid: ", string(length(tt_)),"\n")
    write(f, "Endpoint: ",string(tt_[end]),"\n")
    write(f, "updatescheme: ", string(updatescheme),"\n")
    write(f, "Noise Sigma: ",string(Σobs),"\n")
    write(f, "tuningpars_mcmc: ", string(tp),"\n")
    write(f, "Final value  of rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
    write(f, "Final value of MALA parameter (delta): ",string(δ),"\n")

    write(f, "skip in evaluation of loglikelihood: ",string(sk),"\n")
    write(f, "Average acceptance percentage pCN update steps: ",string(perc_acc_pcn),"\n\n")
    write(f, "Elapsed time: ", string(elapsed),"\n")
    close(f)
end

function write_acc(accinfo,accpcn,nshapes,outdir)
    outdir[end] == "/" && error("provide pathname without trailing '/'")

    # extract number of distinct update steps in accinfo
    nunique = length(unique(map(x->x[1], accinfo)))
    niterates = div(length(accinfo),nunique)
    accdf = DataFrame(kernel = map(x->Symbol(x.kernel), accinfo), acc = map(x->x.acc, accinfo), iter = repeat(1:niterates, inner= nunique))
    accpcndf = DataFrame(kernel = fill(Symbol("pCN"),length(accpcn)), acc=accpcn, iter=repeat(1:niterates,inner=nshapes))
    append!(accdf, accpcndf)
    CSV.write(joinpath(outdir, "accdf.csv"), accdf; delim=";")
end


"""
    Write observations to file
"""
function write_observations(xobs0, xobsT, n, nshapes, outdir)
    outdir[end] == "/" && error("provide pathname without trailing '/'")

    valueT = vcat(map(x->deepvec(x), xobsT)...) # merge all observations at time T in one vector
    if d==1
        posT = repeat(["pos1"], n*nshapes)
    elseif d==2
        posT = repeat(["pos1","pos2"], n*nshapes)
    elseif d==3
        posT = repeat(["pos1","pos2","pos3"], n*nshapes)
    end
    shT = repeat(1:nshapes, inner=d*n)
    obsTdf = DataFrame(pos=posT,shape=shT, value=valueT,landmark=repeat(1:n,inner=d,outer=nshapes))

    if d==1
        obs0df = DataFrame(pos1=extractcomp(xobs0,1),landmark=1:n)
    elseif d==2
        obs0df = DataFrame(pos1=extractcomp(xobs0,1), pos2=extractcomp(xobs0,2),landmark=1:n)
    elseif d==3
        obs0df = DataFrame(pos1=extractcomp(xobs0,1), pos2=extractcomp(xobs0,2), pos3=extractcomp(xobs0,3),landmark=1:n)
    end

    CSV.write(joinpath(outdir, "obs0.csv"), obs0df; delim=";")
    CSV.write(joinpath(outdir, "obsT.csv"), obsTdf; delim=";")
end


"""
    Write parameter iterates to file
"""
function write_params(parsave,subsamples,outdir)
    outdir[end] == "/" && error("provide pathname without trailing '/'")

    parsdf = DataFrame(a=extractcomp(parsave,1),c=extractcomp(parsave,2),
            gamma=extractcomp(parsave,3), iterate=subsamples)
    CSV.write(joinpath(outdir, "parameters.csv"), parsdf; delim=";")
end

"""
    Write noisefields to file (empty in case of MS-model)
"""
function write_noisefields(P,outdir)
    outdir[end] == "/" && error("provide pathname without trailing '/'")

    if isa(P,Landmarks)
        nfsloc = [P.nfs[j].δ for j in eachindex(P.nfs)]
        if d==2
            nfsdf = DataFrame(locx =  extractcomp(nfsloc,1),
                          locy =  extractcomp(nfsloc,2),
                          nfstd=fill(P.nfstd,length(P.nfs)))
        elseif d==1
            nfsdf = DataFrame(locx =  extractcomp(nfsloc,1),
                          nfstd=fill(P.nfstd,length(P.nfs)))
        end
    elseif isa(P,MarslandShardlow)
        nfsdf =DataFrame(locx=Int64[], locy=Int64[], nfstd=Int64[])
    end
    CSV.write(joinpath(outdir, "noisefields.csv"), nfsdf; delim=";")
end
