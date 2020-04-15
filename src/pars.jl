abstract type Pars end

"""
    struct summarising settings for updates on mcmc algorithm. This includes

- `ρinit`
Initial value for ρ, this is a a number in [0,1). Wiener innovations are updated according to
    W_new = ρinit * W_old + sqrt(1-ρinit^2) * W_ind
where W_ind is an independnet Wiener process.
Note that the value of ρinit is adapted during mcmc-iterations.

- `maxnrpaths`
Upper bound on number of landmark paths that is updated in one block.

- `δinit`
Initial value for δ, which is a tuning parameter for updating the initial state.
This is a length-2 vector, where δ[1] is used for tuning updates on the initial positions and
δ[2] is used for tuning updates on the initial momenta.
Note that the value of δ is adapted during mcmc-iterations.

- `covθprop`. For the parameter θ = (a,c,γ) we use Random-Walk updates on log(θ), according to
log(θᵒ) = log(θ) + N(0, covθprop)

- `η``
Stepsize cooling function for the adaptation to ensure diminishing adaptation.

- `adaptskip`
To define guided proposals, we need to plug-in a momentum vector at time T. This is unobserved, and by default initialised
as the zero vector. Now every `adaptskip` iterations, `mT` is updated to the value obtained in that iteration.
"""
@with_kw struct Pars_ms <: Pars
    model:: Symbol = :ms
    ρinit::Float64      = 0.9        # initial value for ρ
    maxnrpaths::Int64   = 10        # at most update maxnrpaths in the innovation-updating step
    covθprop::Array{Float64,2}  = [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04]                 # RW-covariance on Normally distributed update on log(θ)
    η::Any = n -> min(0.2, 10/n)                          # stepsize cooling function
    adaptskip::Int64  =20          # adapt mT (used in Paux) every adaptskip number of iterations
    σobs :: Float64 = 0.01
    dt :: Float64 =0.01
    cinit :: Float64 = 0.2
    γinit :: Float64 = 0.1
    κ :: Float64 = 100.0 # in variance of prior on initial momenta
    δpos :: Float64 = 0.001
    δmom :: Float64 = 0.1
    skip_saveITER :: Int64 = 10 # skip every skip_saveITER iteration in saving paths
end

@with_kw struct Pars_ahs <: Pars
    model:: Symbol = :ahs
    ρinit::Float64      = 0.9        # initial value for ρ
    maxnrpaths::Int64   = 10        # at most update maxnrpaths in the innovation-updating step
    covθprop::Array{Float64,2}  = [0.04 0. 0.; 0. 0.04 0.; 0. 0. 0.04]                 # RW-covariance on Normally distributed update on log(θ)
    η::Any = n -> min(0.2, 10/n)                          # stepsize cooling function
    adaptskip::Int64  =20          # adapt mT (used in Paux) every adaptskip number of iterations
    σobs :: Float64 = 0.01
    dt :: Float64 = 0.01
    cinit :: Float64 = 0.02
    γinit :: Float64 = 0.1
    stdev :: Float64 = 0.75
    db :: Float64 = 2.5
    κ :: Float64 = 100.0 # in variance of prior on initial momenta    δinit_ms ::Array{Float64,1} = [0.001, 0.1]
    δpos :: Float64 = 0.1
    δmom :: Float64 = 0.1
    skip_saveITER :: Int64 = 10 # skip every skip_saveITER iteration in saving paths
end
