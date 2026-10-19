# SelectiveSampling.jl

Decide the obvious cases of an expensive classifier with a cheap surrogate
that learns online, and ask the expensive one only when it matters.

## The problem

Samples `x_1, x_2, …` (feature vectors) arrive in order. An *oracle* `f`
-- expensive, possibly noisy, e.g. an LLM judgement -- returns a score
`y_t = f(x_t)` on request; the sample is *positive* iff `y_t > threshold`.
For every sample the learner either

- **queries** the oracle, pays for it, uses its answer, and learns from it, or
- **abstains** and uses the surrogate's prediction, observing nothing.

The goal is to agree with the oracle as often as possible while querying
it as rarely as possible. The surrogate can at best match the oracle, never
beat it, and since positives are typically rare, a missed positive should
cost more than a wasted query. The surrogate starts untrained and only ever
learns from the samples it chose to query, which biases its training data
towards the cases it was unsure about. Random exploration (`Explore`)
counters that.

This is *stream-based selective sampling*, a.k.a. online active learning
with abstention.

## Quick start

```julia
using SelectiveSampling

model   = BayesianLinear(d)                                  # surrogate
policy  = WarmUp(200, Explore(0.02, Band(lo = 0.02)))         # when to ask
sampler = SelectiveSampler(model, policy; threshold = τ)

for x in stream
    d = decide(sampler, x)
    if d.query
        y = oracle(x)
        observe!(sampler, x, y, d)
        positive = y > τ
    else
        positive = d.positive
    end
end

sampler.stats      # queries made, and unbiased estimates of what was missed
```

Before paying for any oracle calls, tune the policy on data the oracle has
already labelled:

```julia
r = replay(SelectiveSampler(BayesianLinear(d), policy; threshold = τ), X, y)
metrics(r)   # query_rate, recall, precision, missed, est_missed, …
```

## Surrogate models

Online Bayesian linear models with Gaussian posterior `w ~ N(μ, Σ)`, one
rank-1 update per label:

| constructor | oracle | likelihood |
|---|---|---|
| `BayesianLinear(d; …)` | real-valued | Gaussian, noise variance learned (Normal-Inverse-Gamma) |
| `BayesianProbit(d; …)` | binary | probit, moment-matched (ADF) |

Both take `covariance = :full | :diagonal | :scalar`:

| covariance | memory | predict / update | at `d = 4096` |
|---|---|---|---|
| `:full` | `O(d²)` | exact posterior | 1.3 ms / 2.6 ms, 128 MB |
| `:diagonal` | `O(d)` | ignores feature correlations | 2 µs / 3 µs |
| `:scalar` | `O(1)` | one shared variance | 2 µs / 2 µs |

`predict(model, X)` on a matrix scores the columns together; at `d = 4096`
with full covariance a group of 40 takes about 4 ms. Further options:
`forget < 1` for drifting streams (the covariance relaxes towards the
prior, so old evidence fades but uncertainty stays bounded), `τ²` for the
prior weight variance, `intercept` and `intercept_mean`.

Every `Prediction` separates the *epistemic* variance (shrinks with more
labels) from the oracle's *noise* variance (does not), and gives
`p_positive`, the probability that the oracle would label the sample
positive.

## Policies

A policy maps a `Prediction` to a query probability; the sampler then flips
a coin. Randomized decisions are what make the cost estimates in
`sampler.stats` unbiased.

| policy | queries |
|---|---|
| `Band(lo, hi)` | iff `lo ≤ p_positive ≤ hi` -- with `hi = 1`, every possible positive |
| `EpistemicStd(s)` | iff the surrogate's epistemic std exceeds `s` |
| `Margin(b)` | with probability `b / (b + |margin|)` (Cesa-Bianchi et al.) |
| `TargetRate(r)` | a `Band` whose `lo` adapts to query a fraction `r` of decisions |
| `Explore(ε, p)` | with probability at least `ε`, otherwise as `p` |
| `WarmUp(n, p)` | the first `n` decisions, then as `p` |
| `a | b`, `AnyOf` | if any of them would |
| `Always()`, `Never()` | baselines |

Custom policies subtype `Policy` and implement `query_probability(policy,
prediction)` (and `record!(policy, queried)` if they keep state).

## Groups

When one oracle call labels several samples at once, `decide(sampler, X)`
on a matrix decides for its columns jointly: the group is queried with the
largest probability of any member, and `observe!(sampler, X, ys, d)` learns
from all of them.

## Cost estimates

`sampler.stats.est_missed` estimates how many of the samples the surrogate
decided alone were positives it labelled negative; `est_false_positives`
the reverse. Every queried sample on which the surrogate would have been
wrong stands for `(1 - p)/p` unqueried ones (Horvitz-Thompson). The
estimates are unbiased when every sample has `p > 0` -- hence `Explore`.

## References

- N. Cesa-Bianchi, C. Gentile, L. Zaniboni. *Worst-case analysis of
  selective sampling for linear classification.* JMLR 7, 2006.
- A. Beygelzimer, S. Dasgupta, J. Langford. *Importance weighted active
  learning.* ICML 2009.
- T. Graepel, J. Quiñonero Candela, T. Borchert, R. Herbrich.
  *Web-scale Bayesian click-through rate prediction for sponsored search
  advertising in Microsoft's Bing search engine.* ICML 2010 (online probit
  regression by moment matching).
- L. Chen, M. Zaharia, J. Zou. *FrugalGPT: How to use large language models
  while reducing cost and improving performance.* 2023 (LLM cascades).
