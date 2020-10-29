abstract type Pars end

"""
    struct summarising settings for updates on mcmc algorithm.

Besides initial values for (c,γ), the values that are most likely to require adjustments are `covθprop`, `δpos`, `δmom`

- `ρinit`: initial value for ρ, this is a a number in [0,1). Wiener innovations are updated according to
    W_new = ρinit * W_old + sqrt(1-ρinit^2) * W_ind
where W_ind is an independnet Wiener process. Note that the value of ρinit is adapted during mcmc-iterations.

- `covθprop`. For the parameter θ = (a,c,γ) we use Random-Walk updates on log(θ), according to
log(θᵒ) = log(θ) + N(0, covθprop)

- `η`: Stepsize cooling function for the adaptation to ensure diminishing adaptation.
Set at n -> min(0.2, 10/n)

- `adaptskip`: to define guided proposals, we need to plug-in a momentum vector at time T. This is unobserved, and by default initialised as the zero vector. Now every `adaptskip` iterations, `mT` is updated to the value obtained in that iteration. In addition, every `adaptskip` iterations, adaptive tuning of mcmc
parameters is performed.

- `σobs`: If covariance matrix of extrinsic noise is not specified, then it is set to σobs^2*I

- `dt`: mesh width using for discretisation (the mesh always gets transformed by a time-change s  -> s*(2-s))

- `cinit`, : initial value for `c`. Note that we assume Hamltonain kernel x -> c exp(-|x|^2/(2a^2))

- `γinit`: initial value for `gamma` which quantifies the amount of intrinsic noise.

- `aprior`: prior on `a`

- `cprior`: prior on `c`

- `γprior`: prior on `γ`

- `κ`: in case of landmarksmatching, κ quantifies diffusiveness of the prior on the initial momenta

- `δpos`: stepsize for updating intial positions. Note that the value of δpos is adapted during mcmc-iterations.

- `δmom`: stepsize for updating intial momenta. Note that the value of δmom is adapted during mcmc-iterations.

- `skip_saveITER`: skip every skip_saveITER iteration in saving paths
"""
@with_kw struct Pars_ms <: Pars
    model:: Symbol = :ms
    ρinit::Float64 = 0.95
    η::Any = n -> min(0.2, 10/n)
    adaptskip::Int64 = 20
    σobs:: Float64 = 0.01
    dt:: Float64 = 0.01
    aprior:: Exponential{Float64} = Exponential(1.0)
    γprior:: Exponential{Float64} = Exponential(1.0)
    cinit::  Float64 = 1.0
    γinit::  Float64 = 0.1
    κ:: Float64 = 100.0
    δpos:: Float64 = 0.01
    δmom:: Float64 = 0.01
    δa:: Float64 = 0.1
    δγ:: Float64 = 0.0
    skip_saveITER:: Int64 = 10
end


"""
    struct summarising settings for updates on mcmc algorithm.

Besides initial values for (c,γ), the values that are most likely to require adjustments are `covθprop`, `δpos`, `δmom`, `stdev`, `db`

- `ρinit`: initial value for ρ, this is a a number in [0,1). Wiener innovations are updated according to
    W_new = ρinit * W_old + sqrt(1-ρinit^2) * W_ind
where W_ind is an independnet Wiener process. Note that the value of ρinit is adapted during mcmc-iterations.

- `covθprop`. For the parameter θ = (a,c,γ) we use Random-Walk updates on log(θ), according to
log(θᵒ) = log(θ) + N(0, covθprop)

- `η`: Stepsize cooling function for the adaptation to ensure diminishing adaptation.
Set at n -> min(0.2, 10/n)

- `adaptskip`: to define guided proposals, we need to plug-in a momentum vector at time T. This is unobserved, and by default initialised as the zero vector. Now every `adaptskip` iterations, `mT` is updated to the value obtained in that iteration. In addition, every `adaptskip` iterations, adaptive tuning of mcmc
parameters is performed.

- `σobs`: If covariance matrix of extrinsic noise is not specified, then it is set to σobs^2*I

- `dt`: mesh width using for discretisation (the mesh always gets transformed by a time-change s  -> s*(2-s))

- `cinit`, : initial value for `c`. Note that we assume Hamltonain kernel x -> c exp(-|x|^2/(2a^2))

- `γinit`: initial value for `gamma` which quantifies the amount of intrinsic noise.

- `aprior`: prior on `a`

- `cprior`: prior on `c`

- `γprior`: prior on `γ`

- `stdev`: For the AHS-model we take noise fields centred at points that are both horizontally and vertically separaeted by a distance  that is an integer multiple of 2`stdev`

- `db`: assign noisefields within the box [-db[1], db[1]] x [-db[2], db[2]] (and similarly for other dimensions than 2)

- `κ`: in case of landmarksmatching, κ quantifies diffusiveness of the prior on the initial momenta

- `δpos`: stepsize for updating intial positions. Note that the value of δpos is adapted during mcmc-iterations.

- `δmom`: stepsize for updating intial momenta. Note that the value of δmom is adapted during mcmc-iterations.

- `skip_saveITER`: skip every skip_saveITER iteration in saving paths
"""
@with_kw struct Pars_ahs <: Pars
    model:: Symbol = :ahs
    ρinit::Float64  = 0.95
    η::Any = n -> min(0.2, 10/n)
    adaptskip::Int64 = 20
    σobs:: Float64 = 0.01
    dt:: Float64 = 0.01
    aprior:: Exponential{Float64} = Exponential(1.0)
    γprior:: Exponential{Float64} = Exponential(1.0)
    cinit::  Float64 = 1.0
    γinit::  Float64 = 0.1
    stdev:: Float64 = 0.75
    db::Array{Float64,1} = [2.0, 2.0]
    κ:: Float64 = 100.0
    δpos:: Float64 = 0.01
    δmom:: Float64 = 0.01
    δa:: Float64 = 0.1
    δγ:: Float64 = 0.0
    skip_saveITER:: Int64 = 10
end
