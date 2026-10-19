"""
    Policy

Decides how likely the oracle is to be queried for a sample, given the
surrogate's `Prediction`. A policy implements

- `query_probability(policy, prediction) -> p ∈ [0, 1]`, and optionally
- `record!(policy, queried::Bool)`, called after every decision, for
  policies with state (default: no-op).

The `SelectiveSampler` queries with probability `p` by a coin flip rather
than thresholding, which is what makes its cost estimates unbiased (see
`SelectiveSampler`). Policies combine with `|` (query if any of them
would, `AnyOf`) and `Explore` (a floor on `p`).
"""
abstract type Policy end

function query_probability end
record!(::Policy, ::Bool) = nothing

"Query every sample (the oracle-only baseline)."
struct Always <: Policy end
query_probability(::Always, ::Prediction) = 1.0

"Never query (the surrogate-only baseline, or a base for `Explore`)."
struct Never <: Policy end
query_probability(::Never, ::Prediction) = 0.0

"""
    Band(; lo = 0.05, hi = 1.0)

Query iff `lo ≤ p_positive ≤ hi`. Samples confidently negative
(`p_positive < lo`) are left to the surrogate; with `hi < 1`, so are the
confidently positive ones. The default `hi = 1` sends every possible
positive to the oracle -- appropriate when positives are rare and false
positives are costly.
"""
struct Band <: Policy
    lo::Float64
    hi::Float64
end

function Band(; lo::Real = 0.05, hi::Real = 1.0)
    0 <= lo <= hi <= 1 || throw(ArgumentError("need 0 ≤ lo ≤ hi ≤ 1"))
    Band(lo, hi)
end

query_probability(p::Band, pred::Prediction) = p.lo <= pred.p_positive <= p.hi ? 1.0 : 0.0

"""
    EpistemicStd(max_std)

Query iff the epistemic standard deviation of the prediction exceeds
`max_std`, i.e. whenever the surrogate has not seen enough similar samples,
regardless of what it predicts. In the units of the oracle's output (or
of the latent score for `Probit`).
"""
struct EpistemicStd <: Policy
    max_std::Float64
end

query_probability(p::EpistemicStd, pred::Prediction) = sqrt(pred.epistemic_var) > p.max_std ? 1.0 : 0.0

"""
    Margin(b)

The randomized margin rule of Cesa-Bianchi, Gentile & Zaniboni (2006):
query with probability `b / (b + |margin|)`. Samples near the decision
boundary are almost always queried, far ones rarely but never with
probability zero. `b` is in the units of the margin.
"""
struct Margin <: Policy
    b::Float64
    function Margin(b::Real)
        b > 0 || throw(ArgumentError("b must be positive"))
        new(b)
    end
end

query_probability(p::Margin, pred::Prediction) = p.b / (p.b + abs(pred.margin))

"""
    Explore(ε, policy)

Query with probability at least `ε`, otherwise as `policy` would. The
random queries in the region the policy skips are what keep the surrogate
honest there: they correct confident mistakes it would otherwise never
learn about, and give `SelectiveSampler` the samples it needs to estimate
how many positives the skipped samples contained.
"""
struct Explore{P<:Policy} <: Policy
    ε::Float64
    policy::P
    function Explore(ε::Real, policy::P) where {P<:Policy}
        0 <= ε <= 1 || throw(ArgumentError("ε must be in [0, 1]"))
        new{P}(ε, policy)
    end
end

query_probability(p::Explore, pred::Prediction) = max(p.ε, query_probability(p.policy, pred))
record!(p::Explore, q::Bool) = record!(p.policy, q)

"""
    AnyOf(policies...)
    a | b

Query with the largest probability any of the policies gives.
"""
struct AnyOf{P<:Tuple} <: Policy
    policies::P
end

AnyOf(policies::Policy...) = AnyOf(policies)
Base.:|(a::Policy, b::Policy) = AnyOf((a, b))
Base.:|(a::AnyOf, b::Policy) = AnyOf((a.policies..., b))

query_probability(p::AnyOf, pred::Prediction) = maximum(q -> query_probability(q, pred), p.policies)
record!(p::AnyOf, q::Bool) = foreach(r -> record!(r, q), p.policies)

"""
    TargetRate(rate; lo = 0.05, hi = 1.0, η = 0.01)

A `Band` whose lower bound adapts to spend a fixed budget: after every
decision, `lo ← lo · exp(η (queried - rate))`, so `lo` rises while the
sampler queries more than `rate` of the decisions and falls while it
queries less (stochastic approximation). The current bound is `policy.lo`.
"""
mutable struct TargetRate <: Policy
    rate::Float64
    lo::Float64
    hi::Float64
    η::Float64
end

function TargetRate(rate::Real; lo::Real = 0.05, hi::Real = 1.0, η::Real = 0.01)
    0 < rate < 1 || throw(ArgumentError("rate must be in (0, 1)"))
    0 < lo <= hi <= 1 || throw(ArgumentError("need 0 < lo ≤ hi ≤ 1"))
    TargetRate(rate, lo, hi, η)
end

query_probability(p::TargetRate, pred::Prediction) = p.lo <= pred.p_positive <= p.hi ? 1.0 : 0.0
record!(p::TargetRate, q::Bool) = (p.lo = clamp(p.lo * exp(p.η * (q - p.rate)), 1e-9, p.hi); nothing)

"""
    WarmUp(n, policy)

Query the first `n` decisions unconditionally, then follow `policy`. An
untrained surrogate's uncertainty is only as good as its prior; a short
warm-up gives it real labels before it is trusted to skip anything.
"""
mutable struct WarmUp{P<:Policy} <: Policy
    n::Int
    seen::Int
    policy::P
end

WarmUp(n::Integer, policy::Policy) = WarmUp(Int(n), 0, policy)

query_probability(p::WarmUp, pred::Prediction) = p.seen < p.n ? 1.0 : query_probability(p.policy, pred)

function record!(p::WarmUp, q::Bool)
    p.seen < p.n ? (p.seen += 1) : record!(p.policy, q)
    return nothing
end
