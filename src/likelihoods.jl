"""
    Likelihood

Observation model of an `OnlineBayes` model: how the oracle's output `y`
relates to the latent score `wᵀx`.

- `Gaussian`: real-valued `y = wᵀx + ε`, `ε ~ N(0, σ²)`, with `σ²` learned
  online unless fixed.
- `Probit`: binary `P(y = 1) = Φ(wᵀx / β)`.
"""
abstract type Likelihood end

"""
    Gaussian(; σ² = 1.0, learn_noise = true, strength = 1.0)

Real-valued likelihood with noise variance `σ²`. With `learn_noise`, `σ²`
gets a conjugate inverse-gamma prior with mean `σ²` worth `2 * strength`
observations, so the weights and the noise are learned jointly
(Normal-Inverse-Gamma model). The prior on the weights is then expressed
relative to the noise: `w ~ N(μ₀, σ² τ² I)`.
"""
mutable struct Gaussian <: Likelihood
    shape::Float64
    rate::Float64
    shape0::Float64
    rate0::Float64
    learn_noise::Bool
end

function Gaussian(; σ²::Real = 1.0, learn_noise::Bool = true, strength::Real = 1.0)
    σ² > 0 || throw(ArgumentError("σ² must be positive"))
    strength > 0 || throw(ArgumentError("strength must be positive"))
    Gaussian(strength, strength * σ², strength, strength * σ², learn_noise)
end

"""
    noise_var(likelihood)
    noise_var(model)

Current estimate of the oracle's noise variance (`β²` for `Probit`).
"""
noise_var(L::Gaussian) = L.rate / L.shape

"""
    Probit(; β² = 1.0)

Binary likelihood `P(y = 1 | w) = Φ(wᵀx / √β²)`, updated by exact moment
matching (assumed density filtering, as in Bayesian probit regression /
AdPredictor). `β²` is the latent noise; with the default prior it only
sets the scale of the weights.
"""
struct Probit <: Likelihood
    β²::Float64
end

function Probit(; β²::Real = 1.0)
    β² > 0 || throw(ArgumentError("β² must be positive"))
    Probit(β²)
end

noise_var(L::Probit) = L.β²

normcdf(z) = erfc(-z / sqrt(2)) / 2

# φ(t) / Φ(t), stable for all t
mills_inverse(t) = sqrt(2 / π) / erfcx(-t / sqrt(2))

relax!(L::Gaussian, a) = (L.learn_noise && (L.shape = a * L.shape + (1 - a) * L.shape0; L.rate = a * L.rate + (1 - a) * L.rate0); L)
relax!(L::Probit, a) = L
