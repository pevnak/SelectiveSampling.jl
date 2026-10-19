"""
    ReplayResult

Per-sample record of a `replay`:

- `queried`: whether the oracle was asked
- `p`: probability with which that was decided
- `p_positive`: the surrogate's probability of a positive, before the label
- `truth`: the oracle's label (`y > threshold`)
- `predicted`: the label the sampler ended up with -- the oracle's when
  queried, the surrogate's otherwise

plus the sampler's `est_missed` / `est_false_positives` accumulated during
the replay, to compare against the true counts. Summarize with `metrics`.
"""
struct ReplayResult
    queried::BitVector
    p::Vector{Float64}
    p_positive::Vector{Float64}
    truth::BitVector
    predicted::BitVector
    est_missed::Float64
    est_false_positives::Float64
end

"""
    replay(sampler, X, y; groups = nothing) -> ReplayResult

Stream the columns of `X` through `sampler` in order, answering its
queries from the known oracle outputs `y` -- a free simulation of what a
policy would have cost and missed, for tuning it before running it
against the real oracle. `sampler` is updated as it would be live, so pass
a fresh one.

`groups`, if given, is a vector of consecutive column ranges covering
`1:size(X, 2)`, each decided as one `GroupDecision`.
"""
function replay(s::SelectiveSampler, X::AbstractMatrix, y::AbstractVector; groups = nothing)
    n = size(X, 2)
    length(y) == n || throw(DimensionMismatch("$n samples but $(length(y)) labels"))
    groups === nothing && (groups = [j:j for j in 1:n])
    next = 1
    for g in groups
        first(g) == next || throw(ArgumentError("groups must be consecutive ranges covering 1:$n"))
        next = last(g) + 1
    end
    next == n + 1 || throw(ArgumentError("groups must be consecutive ranges covering 1:$n"))

    queried, truth, predicted = falses(n), falses(n), falses(n)
    p, p_positive = zeros(n), zeros(n)
    missed0, fp0 = s.stats.est_missed, s.stats.est_false_positives
    for g in groups
        Xg = view(X, :, g)
        d = decide(s, Xg)
        for (k, j) in enumerate(g)
            queried[j] = d.query
            p[j] = d.p
            p_positive[j] = d.predictions[k].p_positive
            truth[j] = y[j] > s.threshold
            predicted[j] = d.query ? truth[j] : d.positive[k]
        end
        d.query && observe!(s, Xg, view(y, g), d)
    end
    ReplayResult(queried, p, p_positive, truth, predicted,
                 s.stats.est_missed - missed0, s.stats.est_false_positives - fp0)
end

"""
    metrics(r::ReplayResult)

Summary of a replay: number of samples `n`, `query_rate`, number of
`positives`, the `recall` and `precision` of the final labels with respect
to the oracle, the true counts of `missed` positives and `false_positives`,
and the sampler's estimates of the same.
"""
function metrics(r::ReplayResult)
    n = length(r.truth)
    tp = count(r.truth .& r.predicted)
    positives = count(r.truth)
    npred = count(r.predicted)
    (; n,
       query_rate = n == 0 ? NaN : count(r.queried) / n,
       positives,
       recall = positives == 0 ? NaN : tp / positives,
       precision = npred == 0 ? NaN : tp / npred,
       missed = count(r.truth .& .!r.predicted),
       false_positives = count(.!r.truth .& r.predicted),
       est_missed = r.est_missed,
       est_false_positives = r.est_false_positives)
end
