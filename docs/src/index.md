# BridgeLandmarks.jl

```@index
```

```@autodocs
Modules = [BridgeLandmarks]
```

A Julia package for stochastic landmarks matching and template estimation. Inference is based on guided proposals and, more specifically, a simple version of the Backward Filtering Forward Guiding (BFFG) algorithm from <https://arxiv.org/pdf/1712.03807.pdf>. Two stochastic landmarks models are implemented:

- The model by Arnaudon, Holm and Sommer based on stochastic Euler-Poincare equations.
- The model by Trouve and Vialard, which adds a Wiener term to the equation of the momentum of a landmark.  

### Matching of two landmark configurations

### Estimation of a template configuration

### Setting parameters

### Plotting methods (based on R - ggplot)

### Internal details

At each time, the state of the landmarks process is a vector of positions and momenta. Each of such a vector is a `Point{Float64}` (with alias `PointF`) and hence a `State` can be constructed
```julia
st = State(rand(PointF),5), rand(PointF,5))
```



## Example


Compute skeleton graph `h` with separating sets `S` and CPDAG `g` from
the 47x1190 data set NCI-60 on expression profiles of miRNAs and mRNAs.

```julia
using BridgeLandmarks

## Performance



## Contribution
See

## References

* Alain Trouve and Francois-Xavier Vialard. Shape splines and stochastic shape evolutions: A second order point of view. Quarterly of Applied Mathematics, 70(2):219–251, 2012. ISSN 0033-569X, 1552-4485. doi: 10.1090/S0033-569X-2012-01250-4.
* Alexis Arnaudon, Darryl D. Holm, and Stefan Sommer. A Geometric Framework
for Stochastic Shape Analysis. Foundations of Computational Mathematics, 19(3): 653–701, June 2019. ISSN 1615-3383. doi: 10.1007/s10208-018-9394-z.
* M. Mider, M.R. Schauer and F.H. van der Meulen (2020) Continuous-discrete smoothing of diffusions
* A. Arnaudon, F.H. van der Meulen, M.R. Schauer and S. Sommer (2020) Diffusion bridges for stochastic Hamiltonian systems with applications to shape analysis
* J. Bierkens, F.H. van der Meulen and M.R. Schauer (2019) Simulation of elliptic and hypo-elliptic conditional diffusions. To appear in Advances in Applied Probability
