"""
    OnlineBayes

Online Bayesian linear model `y ~ L(wᵀx)` with Gaussian posterior
`w ~ N(μ, Σ)`, where the covariance form `C` (`FullCov`, `DiagCov` or
`ScalarCov`) and the likelihood `L` (`Gaussian` or `Probit`) are chosen
independently. Construct with `BayesianLinear` or `BayesianProbit`.

Every update is one rank-1 step: with `u = Σx`, the mean moves along `u`
and the covariance shrinks by `β u uᵀ`, where the scalars come from the
likelihood. For the Gaussian likelihood `Σ` is kept in units of the noise
variance (Normal-Inverse-Gamma parametrization).

**Forgetting.** With `forget < 1`, before every update the covariance
relaxes towards the prior, `Σ ← forget Σ + (1 - forget) Σ₀`, and the mean
is kept. This is the predict step of a Kalman filter in which the weights
take a random-walk step with covariance `(1 - forget)(Σ₀ - Σ)` (positive
semi-definite, as updates only ever shrink `Σ` below `Σ₀`): old evidence
fades at rate `forget` per update, uncertainty never grows beyond the
prior, and, unlike a mean-reverting walk, the estimate is not biased
towards the prior mean.

The model is not thread-safe: predictions and updates share buffers.
"""
mutable struct OnlineBayes{T<:AbstractFloat, C<:CovarianceForm, L<:Likelihood}
    μ::Vector{T}
    μ0::Vector{T}
    cov::C
    lik::L
    τ²::Float64
    forget::Float64
    intercept::Bool
    nupdates::Int
    x::Vector{T}
    u::Vector{T}
end

function OnlineBayes{T}(d::Integer, lik::Likelihood; covariance::Symbol = :full, τ²::Real = 1.0,
                        forget::Real = 1.0, intercept::Bool = true, intercept_mean::Real = 0.0) where {T}
    d >= 1 || throw(ArgumentError("dimension must be positive"))
    τ² > 0 || throw(ArgumentError("τ² must be positive"))
    0 < forget <= 1 || throw(ArgumentError("forget must be in (0, 1]"))
    D = d + intercept
    μ0 = zeros(T, D)
    intercept && (μ0[end] = intercept_mean)
    C = make_covariance(covariance, T, D, τ²)
    OnlineBayes{T, typeof(C), typeof(lik)}(copy(μ0), μ0, C, lik, Float64(τ²), Float64(forget), intercept, 0,
                                          zeros(T, D), zeros(T, D))
end

"""
    BayesianLinear(d; covariance = :full, σ² = 1.0, learn_noise = true, τ² = 1.0,
                   forget = 1.0, intercept = true, intercept_mean = 0.0, T = Float64)

Online Bayesian linear regression for a real-valued oracle, on `d`
features (plus an intercept, unless `intercept = false`).

- `covariance`: `:full` (exact, `O(d²)`), `:diagonal` or `:scalar` (`O(d)`).
- `σ²`, `learn_noise`: noise variance and whether to learn it online (see
  `Gaussian`).
- `τ²`: prior variance of the weights, *relative to the noise variance*:
  `w ~ N(μ₀, σ² τ² I)`. With standardized features, `τ² = 1` says a feature
  moves the score by about one noise standard deviation.
- `intercept_mean`: prior mean of the intercept, e.g. the expected oracle
  output. The intercept is shrunk towards it.
- `forget`: per-update forgetting factor for drifting streams (see `OnlineBayes`).
"""
function BayesianLinear(d::Integer; σ²::Real = 1.0, learn_noise::Bool = true, strength::Real = 1.0,
                        T::Type{<:AbstractFloat} = Float64, kwargs...)
    OnlineBayes{T}(d, Gaussian(; σ², learn_noise, strength); kwargs...)
end

"""
    BayesianProbit(d; covariance = :full, β² = 1.0, τ² = 1.0, forget = 1.0,
                   intercept = true, intercept_mean = 0.0, T = Float64)

Online Bayesian probit classifier for a binary oracle, on `d` features
(plus an intercept). `τ²` is the absolute prior variance of the weights;
the other arguments are as for `BayesianLinear`. Labels are positive iff
`y > 0.5`, so `Bool`, `0/1` and `±1` codings all work.
"""
function BayesianProbit(d::Integer; β²::Real = 1.0, T::Type{<:AbstractFloat} = Float64, kwargs...)
    OnlineBayes{T}(d, Probit(; β²); kwargs...)
end

"Number of features (excluding the intercept)."
dim(m::OnlineBayes) = length(m.μ) - m.intercept

noise_var(m::OnlineBayes) = noise_var(m.lik)

function Base.show(io::IO, m::OnlineBayes{T, C, L}) where {T, C, L}
    print(io, "OnlineBayes{$T}(", nameof(L), ", ", nameof(C), ", d = ", dim(m),
          m.intercept ? " + intercept" : "", ", updates = ", m.nupdates)
    m.forget < 1 && print(io, ", forget = ", m.forget)
    print(io, ")")
end

function _load!(m::OnlineBayes, x::AbstractVector)
    d = dim(m)
    length(x) == d || throw(DimensionMismatch("expected $d features, got $(length(x))"))
    copyto!(view(m.x, 1:d), x)
    m.intercept && (m.x[end] = 1)
    return m.x
end

# mean and Σ-quadratic form of the loaded features; leaves u = Σx in m.u
function _moments!(m::OnlineBayes)
    cov_mul!(m.u, m.cov, m.x)
    return Float64(dot(m.μ, m.x)), max(Float64(dot(m.x, m.u)), 0.0)
end

"""
    Prediction

What a model knows about one sample:

- `mean`: predicted oracle output (`Gaussian`) or latent score (`Probit`)
- `margin`: `mean - threshold` (`Gaussian`) or `mean` (`Probit`)
- `epistemic_var`: variance of `mean` due to the uncertain weights -- the
  part that more labels reduce
- `noise_var`: the oracle's own noise variance -- the part they don't
- `p_positive`: probability that the oracle labels the sample positive,
  i.e. outputs more than `threshold` (`Gaussian`) or `1` (`Probit`)
"""
struct Prediction
    mean::Float64
    margin::Float64
    epistemic_var::Float64
    noise_var::Float64
    p_positive::Float64
end

_prediction(L::Gaussian, mean, v, threshold) =
    (σ² = noise_var(L); Prediction(mean, mean - threshold, σ² * v, σ², normcdf((mean - threshold) / sqrt(σ² * (1 + v)))))

_prediction(L::Probit, mean, v, threshold) =
    Prediction(mean, mean, v, L.β², normcdf(mean / sqrt(L.β² + v)))

"""
    predict(model, x; threshold = 0.0) -> Prediction

Predictive distribution of the oracle's output for features `x`.
`threshold` defines a positive for `Gaussian` models and is ignored by
`Probit` ones.
"""
function predict(m::OnlineBayes, x::AbstractVector; threshold::Real = 0.0)
    _load!(m, x)
    mean, v = _moments!(m)
    return _prediction(m.lik, mean, v, Float64(threshold))
end

"""
    predict(model, X::AbstractMatrix; threshold = 0.0) -> Vector{Prediction}

Predictions for the columns of `X`, computed together (one matrix-matrix
product for `FullCov`, much faster than predicting column by column).
"""
function predict(m::OnlineBayes{T}, X::AbstractMatrix; threshold::Real = 0.0) where {T}
    d, n = dim(m), size(X, 2)
    size(X, 1) == d || throw(DimensionMismatch("expected $d features, got $(size(X, 1))"))
    Xa = Matrix{T}(undef, length(m.μ), n)
    copyto!(view(Xa, 1:d, :), X)
    m.intercept && (Xa[end, :] .= 1)
    U = similar(Xa)
    cov_mul!(U, m.cov, Xa)
    means = Xa' * m.μ
    return [_prediction(m.lik, Float64(means[j]), max(Float64(dot(view(Xa, :, j), view(U, :, j))), 0.0), Float64(threshold))
            for j in 1:n]
end

function _relax!(m::OnlineBayes)
    a = m.forget
    relax!(m.cov, a, m.τ²)
    relax!(m.lik, a)
    return m
end

"""
    update!(model, x, y; weight = 1.0)

Condition the model on the oracle's answer `y` for features `x`. `weight`
scales the observation's influence (e.g. an importance weight `1/p`);
only the `Gaussian` likelihood supports `weight ≠ 1`.
"""
function update!(m::OnlineBayes, x::AbstractVector, y::Real; weight::Real = 1.0)
    weight > 0 || throw(ArgumentError("weight must be positive"))
    _load!(m, x)
    m.forget < 1 && _relax!(m)
    mean, v = _moments!(m)
    _update!(m, m.lik, mean, v, y, Float64(weight))
    m.nupdates += 1
    return m
end

function _update!(m::OnlineBayes, L::Gaussian, mean, v, y, w)
    r = Float64(y) - mean
    s = 1 / w + v
    axpy!(r / s, m.u, m.μ)
    downdate!(m.cov, m.u, 1 / s)
    if L.learn_noise
        L.shape += w / 2
        L.rate += w * r^2 / (2 * (1 + w * v))
    end
    return m
end

function _update!(m::OnlineBayes, L::Probit, mean, v, y, w)
    w == 1 || throw(ArgumentError("the Probit likelihood does not support weighted updates"))
    sgn = y > 0.5 ? 1.0 : -1.0
    s = sqrt(L.β² + v)
    t = sgn * mean / s
    g = mills_inverse(t)
    axpy!(sgn * g / s, m.u, m.μ)
    downdate!(m.cov, m.u, clamp(g * (g + t), 0.0, 1.0) / s^2)
    return m
end
