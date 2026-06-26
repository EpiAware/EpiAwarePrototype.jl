# Additional infection-process models: exponential growth rate and the renewal
# process, plus the growth-rate / reproduction-number conversion utilities they
# share. Each model subtypes `AbstractEpiAwareModel` and maps a latent path to a
# path of unobserved infections via one `as_turing_model` method.

@doc raw"
Negative moment generating function of a discrete generation interval `w` at
rate `r`: ``\sum_i w_i e^{-r i}``.

# Arguments

  - `r`: the exponential growth rate.
  - `w`: the discrete generation interval weights.

# Examples
```@example neg_MGF
using EpiAwarePrototype
EpiAwarePrototype.neg_MGF(0.1, [0.2, 0.3, 0.5])
```
"
function neg_MGF(r, w::AbstractVector)
    return sum(w[i] * exp(-r * i) for i in 1:length(w))
end

# Derivative of `neg_MGF` with respect to `r`, used by the Newton step in
# `R_to_r`.
function _dneg_MGF_dr(r, w::AbstractVector)
    return -sum(w[i] * i * exp(-r * i) for i in 1:length(w))
end

@doc raw"
Approximate the exponential growth rate `r` implied by a reproduction number
`R₀` and discrete generation interval `w`.

Solves ``R_0 \sum_i w_i e^{-r i} = 1`` by a small-`r` initial guess refined with
`newton_steps` Newton iterations.

# Arguments

  - `R₀`: the reproduction number (or an [`EpiData`](@ref)-bearing model).
  - `w`: the discrete generation interval weights.

# Keyword Arguments

  - `newton_steps`: number of Newton refinement steps (default `2`).
  - `Δd`: generation-interval discretisation width (default `1.0`).

# Examples
```@example R_to_r
using EpiAwarePrototype
R_to_r(1.5, [0.2, 0.3, 0.5])
```
"
function R_to_r(R₀, w::Vector{T}; newton_steps = 2, Δd = 1.0) where {T <: AbstractFloat}
    mean_gen_time = dot(w, 1:length(w)) * Δd
    r_approx = (R₀ - 1) / (R₀ * mean_gen_time)
    for _ in 1:newton_steps
        r_approx -= (R₀ * neg_MGF(r_approx, w) - 1) /
                    (R₀ * _dneg_MGF_dr(r_approx, w))
    end
    return r_approx
end

function R_to_r(R₀, epi_model::AbstractEpiAwareModel; newton_steps = 2, Δd = 1.0)
    return R_to_r(R₀, epi_model.data.gen_int; newton_steps = newton_steps, Δd = Δd)
end

@doc raw"
Reproduction number implied by an exponential growth rate `r` and discrete
generation interval `w`: ``1 / \sum_i w_i e^{-r i}``.

# Arguments

  - `r`: the exponential growth rate.
  - `w`: the discrete generation interval weights.

# Examples
```@example r_to_R
using EpiAwarePrototype
r_to_R(0.1, [0.2, 0.3, 0.5])
```
"
function r_to_R(r, w::AbstractVector)
    return 1 / neg_MGF(r, w)
end

# `exp(y)` written through `LogExpFunctions.xexpy` to match the upstream
# numerics used by `ExpGrowthRate`.
_oneexpy(y::T) where {T} = xexpy(one(T), y)

@doc raw"
Model unobserved infections via a time-varying exponential growth rate.

```math
I_t = g(\hat I_0) \exp\!\left(\sum_{s \le t} r_s\right)
```

where the latent path supplies the log growth rates ``r_s``, ``g`` is
`data.transformation`, and the unconstrained initial infections ``\hat I_0``
come from `initialisation_prior`.

# Arguments

  - `model`: the [`ExpGrowthRate`](@ref) model.
  - `rt`: the latent path of (log) growth rates.

# Examples
```@example ExpGrowthRate
using EpiAwarePrototype, Distributions
data = EpiData([0.2, 0.3, 0.5], exp)
egr = ExpGrowthRate(; data = data, initialisation_prior = Normal())
rand(as_turing_model(egr, randn(10) * 0.05))
```

## Fields

  - `data`: the [`EpiData`](@ref) object.
  - `initialisation_prior`: prior for the unconstrained initial infections.
"
@kwdef struct ExpGrowthRate{S <: Sampleable} <: AbstractEpiAwareModel
    "`EpiData` object."
    data::EpiData
    "Prior for the unconstrained initial infections."
    initialisation_prior::S = Normal()
end

@model function as_turing_model(model::ExpGrowthRate, rt)
    init_incidence ~ model.initialisation_prior
    return _oneexpy.(init_incidence .+ cumsum(rt))
end

@doc raw"
Abstract supertype for renewal accumulation steps (constant generation interval,
with or without susceptible depletion).
"
abstract type AbstractConstantRenewalStep <: AbstractAccumulationStep end

@doc raw"
Renewal step with a constant generation interval (stored reversed).

```math
I_t = R_t \sum_{i=1}^{n-1} I_{t-i} g_i
```
"
struct ConstantRenewalStep{T} <: AbstractConstantRenewalStep
    rev_gen_int::Vector{T}
end

function (recurrent_step::ConstantRenewalStep)(recent_incidence, Rt)
    new_incidence = Rt * dot(recent_incidence, recurrent_step.rev_gen_int)
    return vcat(recent_incidence[2:end], new_incidence)
end

function _renewal_init_state(::ConstantRenewalStep, I₀, r_approx, len_gen_int)
    return I₀ * [exp(-r_approx * t) for t in (len_gen_int - 1):-1:0]
end

get_state(::ConstantRenewalStep, initial_state, state) = last.(state)

@doc raw"
Renewal step with a constant generation interval and a fixed population (with
susceptible depletion).

```math
I_t = \frac{S_{t-1}}{N} R_t \sum_{i=1}^{n-1} I_{t-i} g_i
```
"
struct ConstantRenewalWithPopulationStep{T} <: AbstractConstantRenewalStep
    rev_gen_int::Vector{T}
    pop_size::T
end

function (recurrent_step::ConstantRenewalWithPopulationStep)(
        recent_incidence_and_available_sus, Rt)
    recent_incidence, S = recent_incidence_and_available_sus
    new_incidence = max(S / recurrent_step.pop_size, 1e-6) * Rt *
                    dot(recent_incidence, recurrent_step.rev_gen_int)
    new_S = S - new_incidence
    return [vcat(recent_incidence[2:end], new_incidence), new_S]
end

function _renewal_init_state(
        recurrent_step::ConstantRenewalWithPopulationStep, I₀, r_approx, len_gen_int)
    return [I₀ * [exp(-r_approx * t) for t in (len_gen_int - 1):-1:0],
        recurrent_step.pop_size]
end

function get_state(::ConstantRenewalWithPopulationStep, initial_state, state)
    state .|>
    st -> last(st[1])
end

@doc raw"
Model unobserved infections via a time-varying renewal process.

```math
\mathcal R_t = g(Z_t), \qquad
I_t = \mathcal R_t \sum_{i=1}^{n-1} I_{t-i} g_i
```

where the latent path supplies (log) ``\mathcal R_t``, ``g`` is
`data.transformation`, ``g_i`` is the discrete generation interval, and the
pre-window infections decay at the growth rate implied by ``\mathcal R_1``.

# Arguments

  - `epi_model`: the [`Renewal`](@ref) model.
  - `_Rt`: the latent path of (log) reproduction numbers.

# Examples
```@example Renewal
using EpiAwarePrototype, Distributions
data = EpiData([0.2, 0.3, 0.5], exp)
renewal = Renewal(data; initialisation_prior = Normal())
rand(as_turing_model(renewal, randn(20) * 0.05))
```

## Fields

  - `data`: the [`EpiData`](@ref) object.
  - `initialisation_prior`: prior for the unconstrained initial infections.
  - `recurrent_step`: the renewal accumulation step (an
    [`AbstractConstantRenewalStep`](@ref)).
"
struct Renewal{E <: EpiData, S <: Sampleable, A <: AbstractConstantRenewalStep} <:
       AbstractEpiAwareModel
    "`EpiData` object."
    data::E
    "Prior for the unconstrained initial infections."
    initialisation_prior::S
    "The renewal accumulation step."
    recurrent_step::A
end

function Renewal(data::EpiData; initialisation_prior = Normal())
    recurrent_step = ConstantRenewalStep(reverse(data.gen_int))
    return Renewal(data, initialisation_prior, recurrent_step)
end

function Renewal(; data::EpiData, initialisation_prior = Normal())
    return Renewal(data; initialisation_prior = initialisation_prior)
end

# Initial renewal state from sampled I₀ and R₀, decaying at the implied rate.
function _make_renewal_init(epi_model::Renewal, I₀, Rt₀)
    r_approx = R_to_r(Rt₀, epi_model)
    return _renewal_init_state(
        epi_model.recurrent_step, I₀, r_approx, epi_model.data.len_gen_int)
end

@model function as_turing_model(epi_model::Renewal, _Rt)
    init_incidence ~ epi_model.initialisation_prior
    I₀ = epi_model.data.transformation(init_incidence)
    Rt = epi_model.data.transformation.(_Rt)
    init = _make_renewal_init(epi_model, I₀, Rt[1])
    return accumulate_scan(epi_model.recurrent_step, init, Rt)
end
