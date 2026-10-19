using SelectiveSampling
using SelectiveSampling: FullCov, DiagCov, ScalarCov, variances
using LinearAlgebra, Random, Statistics, Test

@testset "SelectiveSampling" begin

@testset "Gaussian full covariance equals ridge regression" begin
    rng = Xoshiro(1)
    d, n, τ² = 6, 200, 0.5
    X, y = randn(rng, d, n), randn(rng, n)
    m = BayesianLinear(d; τ², intercept = false)
    foreach(j -> update!(m, X[:, j], y[j]), 1:n)
    V = inv(I / τ² + X * X')
    @test m.μ ≈ V * X * y
    @test m.cov.Σ ≈ V
    @test m.nupdates == n
end

@testset "noise variance is learned" begin
    rng = Xoshiro(2)
    d, n = 5, 5000
    w = randn(rng, d)
    m = BayesianLinear(d; σ² = 10.0)
    for _ in 1:n
        x = randn(rng, d)
        update!(m, x, dot(w, x) + 2.0 + 0.3 * randn(rng))
    end
    @test noise_var(m) ≈ 0.09 rtol = 0.1
    @test m.μ[1:d] ≈ w atol = 0.02
    @test m.μ[end] ≈ 2.0 atol = 0.02
    fixed = BayesianLinear(d; σ² = 10.0, learn_noise = false)
    update!(fixed, randn(rng, d), 1.0)
    @test noise_var(fixed) == 10.0
end

@testset "diagonal and scalar forms are exact where they should be" begin
    rng = Xoshiro(3)
    d = 4
    onehot(i) = (x = zeros(d); x[i] = 1; x)
    for make in (BayesianLinear, BayesianProbit)
        full = make(d; intercept = false)
        diag = make(d; covariance = :diagonal, intercept = false)
        for _ in 1:50
            x, y = onehot(rand(rng, 1:d)), randn(rng)
            update!(full, x, y); update!(diag, x, y)
        end
        @test full.μ ≈ diag.μ
        @test variances(full.cov) ≈ variances(diag.cov)
        @test isdiag(full.cov.Σ)
    end
    full = BayesianLinear(1; intercept = false)
    scalar = BayesianLinear(1; covariance = :scalar, intercept = false)
    for _ in 1:50
        x, y = randn(rng, 1), randn(rng)
        update!(full, x, y); update!(scalar, x, y)
    end
    @test full.μ ≈ scalar.μ
    @test full.cov.Σ[1] ≈ scalar.cov.s
end

@testset "all forms learn a linear function" begin
    rng = Xoshiro(4)
    d, n = 10, 3000
    w = randn(rng, d)
    Xtest = randn(rng, d, 500)
    for covariance in (:full, :diagonal, :scalar)
        m = BayesianLinear(d; covariance)
        for _ in 1:n
            x = randn(rng, d)
            update!(m, x, dot(w, x) + 0.1 * randn(rng))
        end
        err = mean(abs2, [predict(m, x).mean for x in eachcol(Xtest)] .- Xtest' * w)
        @test err < 0.05
        @test all(>(0), variances(m.cov))
    end
end

@testset "probit classifier" begin
    rng = Xoshiro(5)
    d, n = 5, 3000
    w = randn(rng, d)
    Xtest = randn(rng, d, 1000)
    truth = Xtest' * w .> 0
    for covariance in (:full, :diagonal, :scalar)
        m = BayesianProbit(d; covariance)
        for _ in 1:n
            x = randn(rng, d)
            update!(m, x, dot(w, x) + 0.3 * randn(rng) > 0)
        end
        acc = mean([predict(m, x).p_positive > 0.5 for x in eachcol(Xtest)] .== truth)
        @test acc > 0.9
    end
    m = BayesianProbit(2)
    @test_throws ArgumentError update!(m, [1.0, 0.0], true; weight = 2.0)
    # extreme margins stay finite
    for _ in 1:20
        update!(m, [100.0, -100.0], false)
    end
    @test all(isfinite, m.μ) && all(isfinite, m.cov.Σ)
end

@testset "forgetting" begin
    rng = Xoshiro(6)
    d = 3
    # no data: uncertainty relaxes to the prior, never beyond
    m = BayesianLinear(d; forget = 0.9, τ² = 2.0, intercept = false)
    for _ in 1:100
        update!(m, [1.0, 0.0, 0.0], 5.0)
    end
    @test m.cov.Σ[2, 2] ≈ 2.0
    @test m.cov.Σ[1, 1] < 2.0
    # tracking a switch
    function track(forget)
        m = BayesianLinear(d; forget, intercept = false)
        w = [1.0, -1.0, 0.5]
        errs = Float64[]
        for t in 1:2000
            t == 1000 && (w = -w)
            x = randn(rng, d)
            t > 1100 && push!(errs, abs2(predict(m, x).mean - dot(w, x)))
            update!(m, x, dot(w, x) + 0.1 * randn(rng))
        end
        mean(errs)
    end
    @test track(0.98) < 0.01
    @test track(1.0) > 0.1
end

@testset "predictions" begin
    m = BayesianLinear(2; σ² = 1.0, learn_noise = false, τ² = 1.0, intercept = false)
    p = predict(m, [1.0, 0.0]; threshold = 0.5)
    @test p.mean == 0 && p.margin == -0.5
    @test p.epistemic_var ≈ 1.0 && p.noise_var ≈ 1.0
    @test p.p_positive ≈ SelectiveSampling.normcdf(-0.5 / sqrt(2))
    @test predict(m, Float16[1, 0]; threshold = 0.5) == p
    @test_throws DimensionMismatch predict(m, [1.0])
    rng = Xoshiro(11)
    for make in (BayesianLinear, BayesianProbit), covariance in (:full, :diagonal, :scalar)
        m = make(4; covariance)
        foreach(_ -> update!(m, randn(rng, 4), rand(rng)), 1:20)
        X = randn(rng, 4, 7)
        batch = predict(m, X; threshold = 0.3)
        single = [predict(m, x; threshold = 0.3) for x in eachcol(X)]
        @test all(f -> getfield.(batch, f) ≈ getfield.(single, f), fieldnames(Prediction))
    end
    @test_throws ArgumentError BayesianLinear(2; covariance = :sparse)
end

@testset "policies" begin
    pred(pp; margin = 0.0, var = 1.0) = Prediction(0.0, margin, var, 1.0, pp)
    @test query_probability(Always(), pred(0.0)) == 1
    @test query_probability(Never(), pred(1.0)) == 0
    b = Band(; lo = 0.1, hi = 0.9)
    @test query_probability.(Ref(b), pred.([0.05, 0.1, 0.5, 0.9, 0.95])) == [0, 1, 1, 1, 0]
    @test query_probability(EpistemicStd(0.5), pred(0.0; var = 0.3)) == 1
    @test query_probability(EpistemicStd(0.5), pred(0.0; var = 0.2)) == 0
    @test query_probability(Margin(2.0), pred(0.0; margin = -2.0)) ≈ 0.5
    @test query_probability(Explore(0.1, Never()), pred(0.0)) == 0.1
    @test query_probability(Explore(0.1, Always()), pred(0.0)) == 1
    any = Band(; lo = 0.5) | EpistemicStd(0.5) | Never()
    @test any isa AnyOf && length(any.policies) == 3
    @test query_probability(any, pred(0.1; var = 1.0)) == 1
    @test query_probability(any, pred(0.1; var = 0.1)) == 0

    w = WarmUp(2, Never())
    @test query_probability(w, pred(0.0)) == 1; record!(w, true)
    @test query_probability(w, pred(0.0)) == 1; record!(w, true)
    @test query_probability(w, pred(0.0)) == 0

    rng = Xoshiro(7)
    tr = TargetRate(0.2; η = 0.05)
    qs = Bool[]
    for _ in 1:20000
        q = query_probability(tr, pred(rand(rng)^4)) == 1
        record!(tr, q)
        push!(qs, q)
    end
    @test mean(qs[end-5000:end]) ≈ 0.2 atol = 0.03
end

@testset "sampler" begin
    rng = Xoshiro(8)
    d = 3
    s = SelectiveSampler(BayesianLinear(d), Explore(0.25, Never()); threshold = 1.0, rng)
    n = 4000
    nq = 0
    for _ in 1:n
        x = randn(rng, d)
        dd = decide(s, x)
        @test dd.p == 0.25
        dd.query && (nq += 1; observe!(s, x, sum(x), dd))
    end
    @test s.stats.n_seen == n && s.stats.n_queried == nq == s.stats.n_labels
    @test nq / n ≈ 0.25 atol = 0.03

    # groups are queried all or nothing, with the largest probability
    s = SelectiveSampler(BayesianLinear(d), Margin(1.0); rng)
    X = [0.0 10.0; 0.0 0.0; 0.0 0.0]
    update!(s.model, [10.0, 0.0, 0.0], 100.0)
    g = decide(s, X)
    @test g.p ≈ maximum(p -> 1 / (1 + abs(p.margin)), g.predictions)
    @test length(g.predictions) == 2 && s.stats.n_seen == 2 && s.stats.n_decisions == 1
    n0 = s.model.nupdates
    observe!(s, X, [0.0, 1.0], g)
    @test s.model.nupdates == n0 + 2

    # probit sampler binarizes the oracle output at the threshold
    s = SelectiveSampler(BayesianProbit(1; intercept = false), Always(); threshold = 2.0)
    for _ in 1:30
        observe!(s, [1.0], 3.0, decide(s, [1.0]))
    end
    @test predict(s.model, [1.0]).p_positive > 0.9
    @test_throws ArgumentError SelectiveSampler(BayesianProbit(1), Always(); importance_weighting = true)
end

@testset "replay and unbiased cost estimates" begin
    d, n = 5, 600
    function run(seed)
        rng = Xoshiro(seed)
        w = randn(rng, d)
        X = randn(rng, d, n)
        y = X' * w .+ 0.5 .* randn(rng, n)
        # a sloppy policy that misses positives, plus exploration
        s = SelectiveSampler(BayesianLinear(d; σ² = 0.25), Explore(0.3, Band(; lo = 0.6)); threshold = 2.0, rng)
        metrics(replay(s, X, y))
    end
    ms = [run(seed) for seed in 1:300]
    @test mean(m.missed for m in ms) > 1
    @test mean(m.est_missed for m in ms) ≈ mean(m.missed for m in ms) rtol = 0.1
    @test mean(m.est_false_positives for m in ms) ≈ mean(m.false_positives for m in ms) rtol = 0.15

    rng = Xoshiro(9)
    X = randn(rng, d, 10)
    s = SelectiveSampler(BayesianLinear(d), Always())
    r = replay(s, X, zeros(10); groups = [1:3, 4:4, 5:10])
    @test all(r.queried) && s.stats.n_decisions == 3
    @test_throws ArgumentError replay(s, X, zeros(10); groups = [1:3, 5:10])
end

@testset "end to end: rare positives" begin
    rng = Xoshiro(10)
    d, n = 20, 20000
    w = randn(rng, d)
    X = randn(rng, d, n)
    y = X' * w .+ 0.5 .* randn(rng, n)
    threshold = quantile(y, 0.97)
    s = SelectiveSampler(BayesianLinear(d), WarmUp(200, Explore(0.02, Band(; lo = 0.02))); threshold, rng)
    m = metrics(replay(s, X, y))
    @test m.recall > 0.97
    @test m.precision > 0.97
    @test m.query_rate < 0.1
    @info "rare positives" m
end

end
