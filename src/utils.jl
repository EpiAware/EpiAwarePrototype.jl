# Backend-agnostic utilities ported from the original `EpiAware` package and
# adapted to the prototype's flattened module layout. None of these depend on
# the `as_turing_model` mechanism; they are plain functions, distributions, and
# accumulation helpers reused across components.

@doc raw"
Apply an [`AbstractAccumulationStep`](@ref) across an input sequence in a single
pass.

This is an optimised `accumulate`-based replacement for an explicit `for` loop.
`acc_step` is a callable step `(state, ϵ) -> new_state`, `initial_state` seeds
the scan, and `ϵ_t` is the driving sequence. The returned value is assembled by
[`get_state`](@ref) from the accumulated states.

# Arguments

  - `acc_step`: an [`AbstractAccumulationStep`](@ref), a callable
    `(state, ϵ) -> new_state` applied at each element of the sequence.
  - `initial_state`: the seed state passed to `accumulate` as `init`.
  - `ϵ_t`: the driving sequence accumulated over.

# Examples
```@example accumulate_scan
using EpiAwarePrototype
accumulate_scan(EpiAwarePrototype.RWStep(), 0.0, [1.0, 2.0, 3.0])
```
"
function accumulate_scan(acc_step::AbstractAccumulationStep, initial_state, ϵ_t)
    result = accumulate(acc_step, ϵ_t; init = initial_state)
    return get_state(acc_step, initial_state, result)
end

@doc raw"
Assemble the final sequence from the raw output of [`accumulate_scan`](@ref).

The default method prepends `initial_state` to the last element of each
accumulated state. Step structs whose state is a named tuple (e.g. the `MA` and
`LatentDelay` steps) override this method to extract the relevant field.

# Arguments

  - `acc_step`: the [`AbstractAccumulationStep`](@ref) used in the scan; the
    method dispatches on its concrete type.
  - `initial_state`: the seed state used by [`accumulate_scan`](@ref).
  - `state`: the raw accumulated output produced by `accumulate`.

# Examples
```@example get_state
using EpiAwarePrototype
accumulate_scan(EpiAwarePrototype.RWStep(), 0.0, [1.0, 2.0, 3.0])
```
"
function get_state(acc_step::AbstractAccumulationStep, initial_state, state)
    return vcat(initial_state, last.(state))
end

@doc raw"
Expand a vector of distributions into a single product distribution.

If every element of `dist` is equal, a `filldist` is returned for efficiency;
otherwise an `arraydist` over the heterogeneous vector is returned.
"
function _expand_dist(dist::Vector{D} where {D <: Distribution})
    d = length(dist)
    product_dist = all(first(dist) .== dist) ?
                   filldist(first(dist), d) : arraydist(dist)
    return product_dist
end

@doc raw"
A half-normal prior distribution parameterised by its mean `μ`.

```math
X \sim \text{HalfNormal}(\mu) \iff X = |Y|,\quad
Y \sim \mathrm{Normal}\!\left(0, \mu\sqrt{\pi/2}\right)
```

so that ``\mathbb{E}[X] = \mu``.

# Examples
```jldoctest HalfNormal; output = false
using EpiAwarePrototype, Distributions
hn = HalfNormal(1.0)
nothing
# output
```
"
struct HalfNormal{T <: Real} <: ContinuousUnivariateDistribution
    μ::T
end

function Base.rand(rng::AbstractRNG, d::HalfNormal{T}) where {T <: Real}
    abs(rand(rng, Normal(0, d.μ * sqrt(π / 2))))
end

function Distributions.logpdf(d::HalfNormal{T}, x::Real) where {T <: Real}
    x < 0 ? -Inf : logpdf(Normal(0, d.μ * sqrt(π / 2)), x) - log(2)
end

function Distributions.cdf(d::HalfNormal{T}, x::Real) where {T <: Real}
    x < 0 ? 0.0 :
    cdf(Normal(0, d.μ * sqrt(π / 2)), x) -
    cdf(Normal(0, d.μ * sqrt(π / 2)), -x)
end

function Distributions.quantile(d::HalfNormal{T}, q::Real) where {T <: Real}
    quantile(Normal(0, d.μ * sqrt(π / 2)), q + (1 - q) / 2)
end

Base.minimum(d::HalfNormal) = 0.0
Base.maximum(d::HalfNormal) = Inf
Distributions.insupport(d::HalfNormal, x::Real) = x >= 0
Statistics.mean(d::HalfNormal{T}) where {T <: Real} = d.μ
Statistics.var(d::HalfNormal{T}) where {T <: Real} = d.μ^2 * (π / 2 - 1)

# --- Safe count distributions ---------------------------------------------
#
# These wrap `Distributions.Poisson` / `NegativeBinomial` to avoid `InexactError`
# at very large means (which can happen mid-sampling) and to declare a
# `SafeIntValued` support so that `eltype` stays integer-typed inside a model.

const SafeInt = Union{Int, BigInt}

@doc raw"
A value-support tag for real-valued count distributions whose `eltype` must stay
integer-typed inside a `Turing` model even when `rand` is called.
"
struct SafeIntValued <: Distributions.ValueSupport end
function Base.eltype(::Type{<:Distributions.Sampleable{F, SafeIntValued}}) where {F}
    SafeInt
end

const SafeDiscreteUnivariateDistribution = Distributions.Distribution{
    Distributions.Univariate, SafeIntValued}

function _safe_int_floor(x::Real)
    Tf = typeof(x)
    if (Tf(typemin(Int)) - one(Tf)) < x < (Tf(typemax(Int)) + one(Tf))
        return floor(Int, x)
    else
        return floor(BigInt, x)
    end
end

@doc raw"
A Poisson distribution parameterised by its mean `λ` that avoids `InexactError`
for very large means.

# Examples
```jldoctest SafePoisson; output = false
using EpiAwarePrototype, Distributions
d = SafePoisson(exp(48.0))
logpdf(d, 100)
nothing
# output
```
"
struct SafePoisson{T <: Real} <: SafeDiscreteUnivariateDistribution
    λ::T

    SafePoisson{T}(λ::Real) where {T <: Real} = new{T}(λ)
    SafePoisson(λ::Real) = SafePoisson{eltype(λ)}(λ)
end

SafePoisson() = SafePoisson{Float64}(1.0)

_poisson(d::SafePoisson) = Poisson(d.λ; check_args = false)

Distributions.params(d::SafePoisson) = _poisson(d) |> params
Distributions.partype(::SafePoisson{T}) where {T} = T
Distributions.rate(d::SafePoisson) = d.λ
Distributions.mean(d::SafePoisson) = d.λ
Distributions.mode(d::SafePoisson) = floor(d.λ)
Distributions.var(d::SafePoisson) = d.λ
Distributions.skewness(d::SafePoisson) = one(typeof(d.λ)) / sqrt(d.λ)
Distributions.kurtosis(d::SafePoisson) = one(typeof(d.λ)) / d.λ
Distributions.logpdf(d::SafePoisson, x::Real) = logpdf(_poisson(d), x)
Distributions.pdf(d::SafePoisson, x::Integer) = pdf(_poisson(d), x)
Distributions.cdf(d::SafePoisson, x::Integer) = cdf(_poisson(d), x)
Distributions.ccdf(d::SafePoisson, x::Integer) = ccdf(_poisson(d), x)
Distributions.quantile(d::SafePoisson, q::Real) = quantile(_poisson(d), q)

Base.minimum(d::SafePoisson) = 0
Base.maximum(d::SafePoisson) = Inf
Distributions.insupport(d::SafePoisson, x::Integer) = x >= 0

# Poisson sampling that stays exact for arbitrarily large means. Small means use
# inversion by exponential interarrivals; large means use the Ahrens-Dieter
# normal-based method. Both floor to an integer via `_safe_int_floor`, which
# promotes to `BigInt` rather than throwing `InexactError` when the mean exceeds
# the `Int` range (which `Distributions.Poisson`'s own sampler does not handle).
#
# Ahrens-Dieter (1982), "Computer Generation of Poisson Deviates from Modified
# Normal Distributions", ACM TOMS 8(2):163-179. Ported from the original
# `EpiAware` package, which adapted it from PoissonRandom.jl.

function _count_rand(rng::AbstractRNG, λ)
    n = 0
    c = randexp(rng)
    while c < λ
        n += 1
        c += randexp(rng)
    end
    return n
end

# log(1+x) - x, accurate over the reduced range used by `_log1pmx`.
function _log1pmx_kernel(x::Float64)
    r = x / (x + 2.0)
    t = r * r
    w = @evalpoly(t, 6.66666666666666667e-1, 4.00000000000000000e-1,
        2.85714285714285714e-1, 2.22222222222222222e-1, 1.81818181818181818e-1,
        1.53846153846153846e-1, 1.33333333333333333e-1, 1.17647058823529412e-1)
    hxsq = 0.5 * x * x
    return r * (hxsq + w * t) - hxsq
end

function _log1pmx(x::Float64)
    if !(-0.7 < x < 0.9)
        return log1p(x) - x
    elseif x > 0.315
        u = (x - 0.5) / 1.5
        return _log1pmx_kernel(u) - 9.45348918918356180e-2 - 0.5 * u
    elseif x > -0.227
        return _log1pmx_kernel(x)
    elseif x > -0.4
        u = (x + 0.25) / 0.75
        return _log1pmx_kernel(u) - 3.76820724517809274e-2 + 0.25 * u
    elseif x > -0.6
        u = (x + 0.5) * 2.0
        return _log1pmx_kernel(u) - 1.93147180559945309e-1 + 0.5 * u
    else
        u = (x + 0.625) / 0.375
        return _log1pmx_kernel(u) - 3.55829253011726237e-1 + 0.625 * u
    end
end

# Procedure F of the Ahrens-Dieter algorithm.
function _procf(λ, K::SafeInt, s::Float64)
    ω = 0.3989422804014327 / s
    b1 = 0.041666666666666664 / λ
    b2 = 0.3 * b1 * b1
    c3 = 0.14285714285714285 * b1 * b2
    c2 = b2 - 15.0 * c3
    c1 = b1 - 6.0 * b2 + 45.0 * c3
    c0 = 1.0 - b1 + 3.0 * b2 - 15.0 * c3
    if K < 10
        px = -float(λ)
        py = λ^K / factorial(K)
    else
        δ = 0.08333333333333333 / K
        δ -= 4.8 * δ^3
        V = (λ - K) / K
        px = K * _log1pmx(V) - δ
        py = 0.3989422804014327 / sqrt(K)
    end
    X = (K - λ + 0.5) / s
    X2 = X^2
    fx = -0.5 * X2
    fy = ω * (((c3 * X2 + c2) * X2 + c1) * X2 + c0)
    return px, py, fx, fy
end

function _ad_rand(rng::AbstractRNG, λ)
    s = sqrt(λ)
    d = 6.0 * λ^2
    L = _safe_int_floor(λ - 1.1484)
    G = λ + s * randn(rng)
    if G >= 0.0
        K = _safe_int_floor(G)
        K >= L && return K
        U = rand(rng)
        d * U >= (λ - K)^3 && return K
        px, py, fx, fy = _procf(λ, K, s)
        fy * (1 - U) <= py * exp(px - fx) && return K
    end
    while true
        E = randexp(rng)
        U = 2.0 * rand(rng) - 1.0
        T = 1.8 + copysign(E, U)
        T <= -0.6744 && continue
        K = _safe_int_floor(λ + s * T)
        px, py, fx, fy = _procf(λ, K, s)
        c = 0.1069 / λ
        @fastmath if c * abs(U) <= py * exp(px + E) - fy * exp(fx + E)
            return K
        end
    end
end

function Base.rand(rng::AbstractRNG, d::SafePoisson)
    d.λ < 6 ? _count_rand(rng, d.λ) : _ad_rand(rng, d.λ)
end

@doc raw"
A negative binomial distribution parameterised by `(r, p)` that avoids
`InexactError` at very large means.

The package uses a mean/cluster-factor parameterisation when constructing this
distribution from an expected count (see
[`NegativeBinomialMeanClust`](@ref)).

# Examples
```jldoctest SafeNegativeBinomial; output = false
using EpiAwarePrototype, Distributions
bigμ = exp(48.0)
σ² = bigμ + 0.05 * bigμ^2
p = bigμ / σ²
r = bigμ * p / (1 - p)
d = SafeNegativeBinomial(r, p)
logpdf(d, 100)
nothing
# output
```
"
struct SafeNegativeBinomial{T <: Real} <: SafeDiscreteUnivariateDistribution
    r::T
    p::T

    SafeNegativeBinomial{T}(r::T, p::T) where {T <: Real} = new{T}(r, p)
end

SafeNegativeBinomial(r::T, p::T) where {T <: Real} = SafeNegativeBinomial{T}(r, p)
SafeNegativeBinomial(r::Real, p::Real) = SafeNegativeBinomial(promote(r, p)...)

_negbin(d::SafeNegativeBinomial) = NegativeBinomial(d.r, d.p; check_args = false)

Base.minimum(d::SafeNegativeBinomial) = 0
Base.maximum(d::SafeNegativeBinomial) = Inf
Distributions.insupport(d::SafeNegativeBinomial, x::Integer) = x >= 0
Distributions.params(d::SafeNegativeBinomial) = _negbin(d) |> params
Distributions.partype(::SafeNegativeBinomial{T}) where {T} = T
Distributions.succprob(d::SafeNegativeBinomial) = _negbin(d).p
Distributions.failprob(d::SafeNegativeBinomial{T}) where {T} = one(T) - _negbin(d).p
Distributions.mean(d::SafeNegativeBinomial) = _negbin(d) |> mean
Distributions.var(d::SafeNegativeBinomial) = _negbin(d) |> var
Distributions.std(d::SafeNegativeBinomial) = _negbin(d) |> std
Distributions.skewness(d::SafeNegativeBinomial) = _negbin(d) |> skewness
Distributions.kurtosis(d::SafeNegativeBinomial) = _negbin(d) |> kurtosis
Distributions.mode(d::SafeNegativeBinomial) = _negbin(d) |> mode
Distributions.logpdf(d::SafeNegativeBinomial, k::Real) = logpdf(_negbin(d), k)
Distributions.cdf(d::SafeNegativeBinomial, x::Real) = cdf(_negbin(d), x)
Distributions.ccdf(d::SafeNegativeBinomial, x::Real) = ccdf(_negbin(d), x)
Distributions.logcdf(d::SafeNegativeBinomial, x::Real) = logcdf(_negbin(d), x)
Distributions.logccdf(d::SafeNegativeBinomial, x::Real) = logccdf(_negbin(d), x)
Distributions.quantile(d::SafeNegativeBinomial, q::Real) = quantile(_negbin(d), q)

function Base.rand(rng::AbstractRNG, d::SafeNegativeBinomial)
    if isone(d.p)
        return 0
    else
        return rand(rng,
            SafePoisson(rand(rng, Gamma(d.r, (1 - d.p) / d.p; check_args = false))))
    end
end

@doc raw"
Construct a [`SafeNegativeBinomial`](@ref) from a mean `μ` and cluster factor
`α` using the variance relationship ``\sigma^2 = \mu + \alpha\,\mu^2``.

# Arguments

  - `μ`: the mean of the distribution.
  - `α`: the cluster factor relating mean and variance.

# Examples
```@example NegativeBinomialMeanClust
using EpiAwarePrototype
NegativeBinomialMeanClust(10.0, 0.1)
```
"
function NegativeBinomialMeanClust(μ, α)
    μ² = μ^2
    σ² = μ + α * μ²
    p = μ / σ²
    r = 1 / α
    return SafeNegativeBinomial(r, p)
end

# Check `censored_cdf` arguments and return the time steps at the right edges of
# the censor intervals. When `D` is `nothing` it is set to the `upper`th quantile
# rounded to a multiple of `Δd`.
function _check_and_give_ts(dist::Distribution, Δd, D, upper)
    @assert minimum(dist)>=0.0 "Distribution must be non-negative."
    @assert Δd>0.0 "Δd must be positive."
    if isnothing(D)
        x_q = invlogcdf(dist, log(upper))
        D = round(Int64, x_q / Δd) * Δd
    end
    @assert D>=Δd "D can't be shorter than Δd."
    ts = Δd:Δd:D |> collect
    @assert ts[end]==D "D must be a multiple of Δd."
    return ts
end

@doc raw"
CDF of ``X + U`` where ``X`` has the CDF of `dist` and ``U`` is uniform on
``[0, Δd)``, evaluated at `t` by quadrature. Used to build the double-censored
CDF and PMF.

# Arguments

  - `dist`: the continuous distribution.
  - `t`: the evaluation point.
  - `Δd`: the censoring-interval width.

# Examples
```@example intF
using EpiAwarePrototype, Distributions
EpiAwarePrototype.∫F(Exponential(1.0), 2.0, 1.0)
```
"
function ∫F(dist, t, Δd)
    return quadgk(u -> exp(logcdf(dist, t - u) - log(Δd)), 0.0, min(Δd, t))[1]
end

@doc raw"
Discrete, double-interval-censored cumulative distribution function of a
continuous distribution.

Assumes a uniform distribution over primary-event times within censoring
intervals of width `Δd`. Returns the (non-truncated) CDF with `0.0` prepended.

# Arguments

  - `dist`: the continuous distribution to discretise.

# Keyword Arguments

  - `Δd`: the censoring-interval width (default `1.0`).
  - `D`: the right-truncation point; `nothing` (default) uses the `upper`th
    quantile rounded to a multiple of `Δd`.
  - `upper`: the quantile used when `D` is `nothing` (default `0.999`).

# Examples
```@example censored_cdf
using EpiAwarePrototype, Distributions
censored_cdf(Exponential(1.0); D = 10)
```
"
function censored_cdf(dist::Distribution; Δd = 1.0, D = nothing, upper = 0.999)
    ts = _check_and_give_ts(dist, Δd, D, upper)
    cens_F = ts .|> t -> ∫F(dist, t, Δd)
    return [0.0; cens_F]
end

@doc raw"
Discrete, right-truncated, double-interval-censored probability mass function of
a continuous distribution.

Differences the double-censored CDF from [`censored_cdf`](@ref) and normalises.
Used to turn a continuous generation interval or reporting delay into the
discrete PMF the models consume.

# Arguments

  - `dist`: the continuous distribution to discretise.

# Keyword Arguments

  - `Δd`: the censoring-interval width (default `1.0`).
  - `D`: the right-truncation point; `nothing` (default) uses the `upper`th
    quantile rounded to a multiple of `Δd`.
  - `upper`: the quantile used when `D` is `nothing` (default `0.99`).

# Examples
```@example censored_pmf
using EpiAwarePrototype, Distributions
censored_pmf(Gamma(2.0, 1.0))
```
"
function censored_pmf(dist::Distribution; Δd = 1.0, D = nothing, upper = 0.99)
    cens_cdf = censored_cdf(dist; Δd, D, upper)
    return cens_cdf |> diff |> p -> p ./ sum(p)
end

@doc raw"
Condition a `DynamicPPL.Model` by fixing some parameters and conditioning on
others.

```julia
condition_model(model, fix_parameters, condition_parameters)
```

equals `condition(fix(model, fix_parameters), condition_parameters)`. Either
named tuple may be empty.

# Arguments

  - `model`: the `DynamicPPL.Model` to fix and condition.
  - `fix_parameters`: a named tuple of parameters to fix to constant values.
  - `condition_parameters`: a named tuple of parameters to condition on data.

# Examples
```@example condition_model
using EpiAwarePrototype, Distributions
m = as_turing_model(RandomWalk(), 10)
condition_model(m, (rw_init = 0.0,), NamedTuple())
```
"
function condition_model(model::DynamicPPL.Model, fix_parameters::NamedTuple,
        condition_parameters::NamedTuple)
    _model = fix(model, fix_parameters)
    _model = condition(_model, condition_parameters)
    return _model
end
