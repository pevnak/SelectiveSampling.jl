"""
    CovarianceForm

Representation of the posterior covariance `Σ` of an `OnlineBayes` model.
Every form supports the same three operations, which is all the model
needs:

- `cov_mul!(u, C, x)`: `u = Σx`
- `downdate!(C, u, β)`: `Σ ← Σ - β u uᵀ` (with `u = Σx`), projected back
  onto the form
- `relax!(C, a, v0)`: `Σ ← aΣ + (1 - a) v0 I`

The projection makes the cheaper forms an assumed-density-filtering
approximation of the full posterior: `DiagCov` keeps the diagonal of the
exactly updated covariance, `ScalarCov` its average diagonal (`tr Σ / d`).
Both are exact when the full covariance would have been diagonal resp. a
multiple of the identity anyway, e.g. for one-hot features.
"""
abstract type CovarianceForm end

"""
    FullCov

Dense `d × d` covariance: exact posterior, `O(d²)` memory and work per
prediction and update.
"""
struct FullCov{T} <: CovarianceForm
    Σ::Matrix{T}
end

FullCov{T}(d::Integer, v::Real) where {T} = FullCov{T}(Matrix{T}(T(v) * I, d, d))

"""
    DiagCov

Diagonal covariance: `O(d)` memory and work, ignores correlations between
features.
"""
struct DiagCov{T} <: CovarianceForm
    σ::Vector{T}
end

DiagCov{T}(d::Integer, v::Real) where {T} = DiagCov{T}(fill(T(v), d))

"""
    ScalarCov

Isotropic covariance `s I`: `O(1)` memory, `O(d)` work. One number tracks
how much has been learned overall.
"""
mutable struct ScalarCov{T} <: CovarianceForm
    s::T
    d::Int
end

ScalarCov{T}(d::Integer, v::Real) where {T} = ScalarCov{T}(T(v), Int(d))

cov_mul!(u, C::FullCov, x) = mul!(u, C.Σ, x)
cov_mul!(u, C::DiagCov, x) = (u .= C.σ .* x)
cov_mul!(u, C::ScalarCov, x) = (u .= C.s .* x)

downdate!(C::FullCov, u, β) = (BLAS.ger!(-oftype(β, β), u, u, C.Σ); C)
downdate!(C::DiagCov, u, β) = (C.σ .-= β .* abs2.(u); C)
downdate!(C::ScalarCov, u, β) = (C.s -= β * sum(abs2, u) / C.d; C)

function relax!(C::FullCov, a, v0)
    lmul!(a, C.Σ)
    for i in axes(C.Σ, 1)
        C.Σ[i, i] += (1 - a) * v0
    end
    return C
end
relax!(C::DiagCov, a, v0) = (C.σ .= a .* C.σ .+ (1 - a) * v0; C)
relax!(C::ScalarCov, a, v0) = (C.s = a * C.s + (1 - a) * v0; C)

"Posterior variances of the individual weights."
variances(C::FullCov) = diag(C.Σ)
variances(C::DiagCov) = copy(C.σ)
variances(C::ScalarCov) = fill(C.s, C.d)

function make_covariance(form::Symbol, ::Type{T}, d::Integer, v::Real) where {T}
    form === :full && return FullCov{T}(d, v)
    form === :diagonal && return DiagCov{T}(d, v)
    form === :scalar && return ScalarCov{T}(d, v)
    throw(ArgumentError("covariance must be :full, :diagonal or :scalar, got :$form"))
end
