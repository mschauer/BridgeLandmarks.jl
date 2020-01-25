@assert d==2
xobs0comp1 = extractcomp(xobs0,1)
xobs0comp2 = extractcomp(xobs0,2)
xobsTcomp1 = extractcomp(xobsT[1],1)
xobsTcomp2 = extractcomp(xobsT[1],2)
pp1 = plotshapes(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2)


anim = @animate for i in 1:ITER
    drawpath(ITER, i, P.n,x,X,objvals,parsave,(xobs0comp1,xobs0comp2,xobsTcomp1, xobsTcomp2),pb)
end

if make_animation
    fn = string(model)
    gif(anim, outdir*"anim.gif", fps = 50)
    mp4(anim, outdir*"anim.mp4", fps = 50)
end

# plotting
#Pdeterm = MarslandShardlow(0.1, 0.1, 0.0, 0.0, P.n)
#plotlandmarkpositions[](initSamplePath(0:0.01:0.1,x0),Pdeterm,x0.q,deepvec2state(xᵒ).q;db=.5)
