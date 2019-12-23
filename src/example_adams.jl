workdir = @__DIR__
cd(workdir)


using Distributions
using PointProcessInference
const PPI = PointProcessInference
using DelimitedFiles


obs = readdlm("adams.txt")

obs = rand(Exponential(2), 100)
res = PPI.inference(obs)

include(PPI.plotscript())
plotposterior(res)
