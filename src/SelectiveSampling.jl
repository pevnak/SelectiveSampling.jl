"""
    SelectiveSampling

Stream-based selective sampling: samples `x_1, x_2, …` arrive in order, an
expensive (possibly noisy) oracle can label any of them on request, and a
cheap online surrogate model learns from the labels obtained so far to
decide the obvious cases itself.

The pieces are independent and composable:

- **Models** (`BayesianLinear`, `BayesianProbit`): online Bayesian linear
  models with full, diagonal or scalar covariance, updated in one rank-1
  step per label, with optional forgetting for drifting streams.
- **Policies** (`Band`, `EpistemicStd`, `Margin`, `Explore`, `TargetRate`,
  `WarmUp`, …): map a `Prediction` to the probability of querying the
  oracle. Combine with `|`.
- **`SelectiveSampler`**: the loop glue -- `decide` whether to query (with
  a coin flip at the policy's probability), `observe!` the oracle's answer,
  and keep unbiased estimates of what the skipped samples cost.
- **`replay`**: run a sampler over an already-labelled stream, to tune
  policies offline before paying for oracle calls.
"""
module SelectiveSampling

using LinearAlgebra
using Random
using SpecialFunctions: erfc, erfcx

include("covariance.jl")
include("likelihoods.jl")
include("model.jl")
include("policies.jl")
include("sampler.jl")
include("replay.jl")

export OnlineBayes, BayesianLinear, BayesianProbit, Prediction, predict, update!, noise_var, dim

export Policy, query_probability, record!,
       Always, Never, Band, EpistemicStd, Margin, Explore, AnyOf, TargetRate, WarmUp

export SelectiveSampler, Decision, GroupDecision, decide, observe!, reset_stats!

export replay, ReplayResult, metrics

end # module SelectiveSampling
