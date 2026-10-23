"""
    SamplerStats

Running counts of a `SelectiveSampler`:

- `n_seen`, `n_queried`: samples decided on, and how many of them were
  sent to the oracle
- `n_decisions`, `n_decisions_queried`: the same per decision (a group of
  samples decided together counts once)
- `n_labels`, `n_label_positive`: labels observed, and how many were positive
- `est_missed`, `est_false_positives`: unbiased estimates of how many of
  the samples the surrogate decided on its own were positives it labelled
  negative, resp. negatives it labelled positive

The estimates use the queried samples only. A queried sample that the
surrogate would have got wrong stands for `(1 - p)/p` unqueried ones,
where `p` is the probability with which it was queried (Horvitz-Thompson).
They are unbiased as long as every sample has `p > 0` -- use `Explore` --
and need many queried samples to be accurate when `p` is small.
"""
mutable struct SamplerStats
    n_seen::Int
    n_queried::Int
    n_decisions::Int
    n_decisions_queried::Int
    n_labels::Int
    n_label_positive::Int
    est_missed::Float64
    est_false_positives::Float64
end

SamplerStats() = SamplerStats(0, 0, 0, 0, 0, 0, 0.0, 0.0)

"""
    SelectiveSampler(model, policy; threshold = 0.0, decision_p = 0.5,
                     importance_weighting = false, rng = Random.default_rng())

Online selective sampling with `model` as the surrogate and `policy`
deciding when to ask the oracle:

```julia
d = decide(sampler, x)
if d.query
    y = oracle(x)
    observe!(sampler, x, y, d)
    label = y > sampler.threshold
else
    label = d.positive
end
```

- `threshold`: the oracle's output is positive iff `y > threshold`.
- `decision_p`: the surrogate labels a sample positive iff its
  `p_positive ≥ decision_p`. Lower it when missing a positive costs more
  than a false alarm.
- `importance_weighting`: weight every queried label by `1/p`, which
  corrects the surrogate's training set for being sampled where it was
  uncertain (Beygelzimer, Dasgupta & Langford, 2009). With a Bayesian
  surrogate and an `Explore` floor this is usually unnecessary, and the
  large weights make updates noisy. `Gaussian` models only.
"""
mutable struct SelectiveSampler{M<:OnlineBayes, P<:Policy, R<:AbstractRNG}
    model::M
    policy::P
    threshold::Float64
    decision_p::Float64
    importance_weighting::Bool
    rng::R
    stats::SamplerStats
end

function SelectiveSampler(model::OnlineBayes, policy::Policy; threshold::Real = 0.0, decision_p::Real = 0.5,
                          importance_weighting::Bool = false, rng::AbstractRNG = Random.default_rng())
    0 < decision_p < 1 || throw(ArgumentError("decision_p must be in (0, 1)"))
    importance_weighting && model.lik isa Probit &&
        throw(ArgumentError("importance weighting needs a Gaussian likelihood"))
    SelectiveSampler(model, policy, Float64(threshold), Float64(decision_p), importance_weighting, rng, SamplerStats())
end

"Reset the sampler's `SamplerStats` (the model and policy are kept)."
reset_stats!(s::SelectiveSampler) = (s.stats = SamplerStats(); s)

"""
    Decision

Result of `decide` for one sample: whether to `query` the oracle, the
probability `p` with which that was decided, the surrogate's `prediction`,
and its own label `positive` (to use when not querying).
"""
struct Decision
    query::Bool
    p::Float64
    prediction::Prediction
    positive::Bool
end

"""
    GroupDecision

Result of `decide` for a group of samples that are queried all together or
not at all (e.g. because one oracle call labels them all). `p` is the
largest query probability in the group, so every member is queried with
probability `p`.
"""
struct GroupDecision
    query::Bool
    p::Float64
    predictions::Vector{Prediction}
    positive::BitVector
end

# A prediction that is not a number (e.g. from non-finite features) means the
# surrogate cannot tell; policies would read it as a confident negative, as
# every comparison with NaN is false. Ask the oracle instead.
function _query_probability(s::SelectiveSampler, pred::Prediction)
    isfinite(pred.p_positive) || return 1.0
    p = Float64(query_probability(s.policy, pred))
    return isnan(p) ? 1.0 : clamp(p, 0.0, 1.0)
end

function _flip!(s::SelectiveSampler, p::Float64, n::Int)
    q = p >= 1 || (p > 0 && rand(s.rng) < p)
    record!(s.policy, q)
    st = s.stats
    st.n_seen += n
    st.n_queried += q * n
    st.n_decisions += 1
    st.n_decisions_queried += q
    return q
end

"""
    decide(sampler, x::AbstractVector) -> Decision
    decide(sampler, X::AbstractMatrix) -> GroupDecision

Predict with the surrogate and decide, by a coin flip with the policy's
probability, whether to query the oracle -- for one sample, or for the
columns of `X` as one group.
"""
function decide(s::SelectiveSampler, x::AbstractVector)
    pred = predict(s.model, x; s.threshold)
    p = _query_probability(s, pred)
    return Decision(_flip!(s, p, 1), p, pred, pred.p_positive >= s.decision_p)
end

function decide(s::SelectiveSampler, X::AbstractMatrix)
    preds = predict(s.model, X; s.threshold)
    p = isempty(preds) ? 0.0 : maximum(pred -> _query_probability(s, pred), preds)
    positive = BitVector([pred.p_positive >= s.decision_p for pred in preds])
    return GroupDecision(_flip!(s, p, length(preds)), p, preds, positive)
end

_target(::Gaussian, y, threshold) = y
_target(::Probit, y, threshold) = y > threshold

function _observe_one!(s::SelectiveSampler, x, y::Real, p::Float64, queried::Bool, surrogate_positive::Bool)
    weight = s.importance_weighting && queried ? 1 / p : 1.0
    update!(s.model, x, _target(s.model.lik, y, s.threshold); weight)
    st = s.stats
    positive = y > s.threshold
    st.n_labels += 1
    st.n_label_positive += positive
    if queried
        r = (1 - p) / p
        positive && !surrogate_positive && (st.est_missed += r)
        !positive && surrogate_positive && (st.est_false_positives += r)
    end
    return s
end

"""
    observe!(sampler, x, y, d::Decision)
    observe!(sampler, X, ys, d::GroupDecision)
    observe!(sampler, x, y)

Update the surrogate with the oracle's output `y` for features `x` (or the
columns of `X`), and the sampler's estimates with what the surrogate would
have decided. Without a decision -- e.g. to warm-start from labels
obtained earlier -- only the model is updated.
"""
observe!(s::SelectiveSampler, x::AbstractVector, y::Real, d::Decision) =
    _observe_one!(s, x, y, d.p, d.query, d.positive)

function observe!(s::SelectiveSampler, X::AbstractMatrix, ys::AbstractVector, d::GroupDecision)
    size(X, 2) == length(ys) == length(d.positive) ||
        throw(DimensionMismatch("$(size(X, 2)) samples, $(length(ys)) labels, $(length(d.positive)) decisions"))
    for (j, x) in enumerate(eachcol(X))
        _observe_one!(s, x, ys[j], d.p, d.query, d.positive[j])
    end
    return s
end

function observe!(s::SelectiveSampler, x::AbstractVector, y::Real)
    update!(s.model, x, _target(s.model.lik, y, s.threshold))
    return s
end
