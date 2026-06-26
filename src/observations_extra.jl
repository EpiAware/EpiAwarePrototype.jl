# Observation-model modifiers and manipulators. Each struct subtypes
# `AbstractEpiAwareModel` and implements one `as_turing_model` method that wraps,
# transforms, aggregates, prefixes, records, or stacks an underlying observation
# model. As with the latent modifiers, prefixing — previously the
# `prefix_submodel` helper — is done here with `DynamicPPL.prefix` applied to the
# inner model before `to_submodel(..., false)`.

# --- ascertainment ----------------------------------------------------------

@doc raw"
Scale the expected observations of an underlying observation model by a latent
ascertainment process.

A latent model generates a length-`length(Y_t)` series which is combined with the
expected observations `Y_t` through `transform` before being passed to the inner
observation `model`. The default `transform` applies a multiplicative effect on
the exponential scale (`(Y_t, x) -> xexpy.(Y_t, x)`), so a latent value `x`
multiplies the expected observation by `exp(x)`. The latent model is wrapped in a
[`PrefixLatentModel`](@ref) (with prefix `latent_prefix`) unless the prefix is the
empty string.

# Arguments

  - `obs_model`: the [`Ascertainment`](@ref) model.
  - `y_t`: the observed series (or `missing` when simulating predictively).
  - `Y_t`: the expected-observation series.

# Examples
```@example Ascertainment
using EpiAwarePrototype, Distributions
obs = Ascertainment(PoissonError(), FixedIntercept(0.1))
mdl = as_turing_model(obs, missing, fill(10.0, 5))
rand(mdl)
```

## Fields

  - `model`: the underlying observation model the ascertained expected
    observations are passed to.
  - `latent_model`: the latent model generating the ascertainment effect
    (prefix-wrapped unless `latent_prefix` is empty).
  - `transform`: the function `(Y_t, x)` combining expected observations with the
    latent effect.
  - `latent_prefix`: the prefix applied to the latent model's variables.
"
struct Ascertainment{
    M <: AbstractEpiAwareModel, L <: AbstractEpiAwareModel, F <: Function,
    P <: String} <: AbstractEpiAwareModel
    "The underlying observation model."
    model::M
    "The latent model generating the ascertainment effect."
    latent_model::L
    "The function combining expected observations with the latent effect."
    transform::F
    "The prefix applied to the latent model's variables."
    latent_prefix::P

    function Ascertainment(model::M, latent_model::L, transform::F,
            latent_prefix::P) where {M <: AbstractEpiAwareModel,
            L <: AbstractEpiAwareModel, F <: Function, P <: String}
        @assert hasmethod(transform, Tuple{Vector, Vector}) "transform must have a method for (Vector, Vector)"
        wrapped_latent_model = latent_prefix == "" ? latent_model :
                               PrefixLatentModel(latent_model, latent_prefix)
        return new{M, typeof(wrapped_latent_model), F, P}(
            model, wrapped_latent_model, transform, latent_prefix)
    end
end

function Ascertainment(model::M, latent_model::L;
        transform = (Y_t, x) -> xexpy.(Y_t, x),
        latent_prefix::String = "Ascertainment") where {
        M <: AbstractEpiAwareModel, L <: AbstractEpiAwareModel}
    return Ascertainment(model, latent_model, transform, latent_prefix)
end

function Ascertainment(; model::M, latent_model::L,
        transform = (Y_t, x) -> xexpy.(Y_t, x),
        latent_prefix::String = "Ascertainment") where {
        M <: AbstractEpiAwareModel, L <: AbstractEpiAwareModel}
    return Ascertainment(model, latent_model, transform, latent_prefix)
end

@model function as_turing_model(obs_model::Ascertainment, y_t, Y_t)
    expected_obs_mod ~ to_submodel(
        as_turing_model(obs_model.latent_model, length(Y_t)), false)
    expected_obs = obs_model.transform(Y_t, expected_obs_mod)
    y_t ~ to_submodel(as_turing_model(obs_model.model, y_t, expected_obs), false)
    return y_t
end

@doc raw"
Build an [`Ascertainment`](@ref) model for a day-of-week reporting effect.

The latent model is wrapped with [`broadcast_dayofweek`](@ref) so a 7-day effect
is broadcast across the expected-observation series, and combined multiplicatively
with the expected observations by default.

# Arguments

  - `model`: the underlying observation model.

# Keyword Arguments

  - `latent_model`: the latent model broadcast over the week (default
    [`HierarchicalNormal`](@ref)`()`).
  - `transform`: the function `(x, y)` combining expected observations with the
    broadcast effect (default `(x, y) -> x .* y`).
  - `latent_prefix`: the prefix applied to the latent model's variables (default
    `\"DayofWeek\"`).

# Examples
```@example ascertainment_dayofweek
using EpiAwarePrototype
obs = ascertainment_dayofweek(PoissonError())
mdl = as_turing_model(obs, missing, fill(10.0, 14))
rand(mdl)
```
"
function ascertainment_dayofweek(model; latent_model = HierarchicalNormal(),
        transform = (x, y) -> x .* y, latent_prefix = "DayofWeek")
    return Ascertainment(
        model, broadcast_dayofweek(latent_model), transform, latent_prefix)
end

# --- aggregation ------------------------------------------------------------

# Scatter the predicted observations for the present time points back into a
# length-`n` vector of expected observations (zeros where not present).
function _return_aggregate(pred_obs, present, n)
    agg_obs = zeros(eltype(pred_obs), n)
    agg_obs[findall(present)] = pred_obs
    return agg_obs
end

@doc raw"
Aggregate the expected observations of an underlying model over reporting windows.

Each entry of `aggregation` gives the window length to sum over at the
corresponding (broadcast) time point, and `present` (derived as
`aggregation .!= 0`) marks the time points that are reported. The aggregation and
presence vectors are broadcast to the observation length with
[`RepeatEach`](@ref), the expected observations are summed over each window, the
inner `model` is applied to the present windows, and the predictions are scattered
back into a full-length vector (zeros elsewhere).

# Arguments

  - `ag`: the [`Aggregate`](@ref) model.
  - `y_t`: the observed series (or `missing` when simulating predictively).
  - `Y_t`: the expected-observation series.

# Examples
```@example Aggregate
using EpiAwarePrototype
obs = Aggregate(PoissonError(), [0, 0, 0, 0, 0, 0, 7])
mdl = as_turing_model(obs, missing, fill(10.0, 14))
rand(mdl)
```

## Fields

  - `model`: the underlying observation model applied to the aggregated windows.
  - `aggregation`: the per-period window lengths (`0` marks an unreported point).
  - `present`: the boolean presence mask (`aggregation .!= 0`).
"
struct Aggregate{
    M <: AbstractEpiAwareModel, A <: AbstractVector{<:Int},
    P <: AbstractVector{<:Bool}} <: AbstractEpiAwareModel
    "The underlying observation model."
    model::M
    "The per-period aggregation window lengths."
    aggregation::A
    "The boolean presence mask."
    present::P

    function Aggregate(model::M, aggregation::A) where {
            M <: AbstractEpiAwareModel, A <: AbstractVector{<:Int}}
        present = aggregation .!= 0
        return new{M, A, typeof(present)}(model, aggregation, present)
    end
end

function Aggregate(; model::M, aggregation::A) where {
        M <: AbstractEpiAwareModel, A <: AbstractVector{<:Int}}
    return Aggregate(model, aggregation)
end

@model function as_turing_model(ag::Aggregate, y_t, Y_t)
    if ismissing(y_t)
        y_t = Vector{Missing}(missing, length(Y_t))
    end
    n = length(y_t)
    m = length(ag.aggregation)
    aggregation = broadcast_rule(RepeatEach(), ag.aggregation, n, m)
    present = broadcast_rule(RepeatEach(), ag.present, n, m)
    agg_Y_t = map(findall(present)) do i
        sum(Y_t[max(1, i - aggregation[i] + 1):i])
    end
    pred_obs ~ to_submodel(
        as_turing_model(ag.model, y_t[present], agg_Y_t), false)
    return _return_aggregate(pred_obs, present, n)
end

# --- prefixing --------------------------------------------------------------

@doc raw"
Wrap an inner observation model so its sampled variables are prefixed with
`prefix`.

This replaces the original `prefix_submodel` helper for observation models: the
inner model is prefixed with `DynamicPPL.prefix` before being sampled as a
submodel, so its variables appear as `prefix.varname`.

# Arguments

  - `observation_model`: the [`PrefixObservationModel`](@ref).
  - `y_t`: the observed series (or `missing` when simulating predictively).
  - `Y_t`: the expected-observation series.

# Examples
```@example PrefixObservationModel
using EpiAwarePrototype
pm = PrefixObservationModel(; model = PoissonError(), prefix = \"Test\")
mdl = as_turing_model(pm, missing, fill(10.0, 5))
rand(mdl)
```

## Fields

  - `model`: the inner observation model to prefix.
  - `prefix`: the string prefix applied to the inner model's variables.
"
@kwdef struct PrefixObservationModel{M <: AbstractEpiAwareModel, P <: String} <:
              AbstractEpiAwareModel
    "The observation model."
    model::M
    "The prefix for the observation model."
    prefix::P
end

@model function as_turing_model(observation_model::PrefixObservationModel, y_t, Y_t)
    submodel ~ to_submodel(
        prefix(as_turing_model(observation_model.model, y_t, Y_t),
            Symbol(observation_model.prefix)), false)
    return submodel
end

# --- recording --------------------------------------------------------------

@doc raw"
Record the expected observations as a tracked generated quantity (`exp_y_t`).

The expected observations `Y_t` are tracked via the `:=` syntax before the inner
`model` is applied unchanged, so the expected observations are available in the
returned chain alongside the inner model's variables.

# Arguments

  - `model`: the [`RecordExpectedObs`](@ref) model.
  - `y_t`: the observed series (or `missing` when simulating predictively).
  - `Y_t`: the expected-observation series.

# Examples
```@example RecordExpectedObs
using EpiAwarePrototype
obs = RecordExpectedObs(PoissonError())
mdl = as_turing_model(obs, missing, fill(10.0, 5))
rand(mdl)
```

## Fields

  - `model`: the inner observation model whose expected observations are recorded.
"
struct RecordExpectedObs{M <: AbstractEpiAwareModel} <: AbstractEpiAwareModel
    "The inner observation model whose expected observations are recorded."
    model::M
end

@model function as_turing_model(model::RecordExpectedObs, y_t, Y_t)
    exp_y_t := Y_t
    y_t ~ to_submodel(as_turing_model(model.model, y_t, Y_t), false)
    return y_t
end

# --- transformation ---------------------------------------------------------

@doc raw"
Apply a transformation function to the expected observations before passing them
to an inner observation model.

The expected observations `Y_t` are mapped through `transform` and the result is
passed to the inner `model`. The default `transform` applies a softplus
(`x -> log1pexp.(x)`), keeping the transformed expected observations positive.

# Arguments

  - `obs`: the [`TransformObservationModel`](@ref).
  - `y_t`: the observed series (or `missing` when simulating predictively).
  - `Y_t`: the expected-observation series.

# Examples
```@example TransformObservationModel
using EpiAwarePrototype
obs = TransformObservationModel(PoissonError(), x -> x .* 2)
mdl = as_turing_model(obs, missing, fill(10.0, 5))
rand(mdl)
```

## Fields

  - `model`: the inner observation model the transformed expected observations are
    passed to.
  - `transform`: the transformation applied to the expected observations.
"
@kwdef struct TransformObservationModel{M <: AbstractEpiAwareModel, F <: Function} <:
              AbstractEpiAwareModel
    "The inner observation model."
    model::M
    "The transformation applied to the expected observations."
    transform::F = x -> log1pexp.(x)
end

function TransformObservationModel(model::M; transform = x -> log1pexp.(x)) where {
        M <: AbstractEpiAwareModel}
    return TransformObservationModel(model, transform)
end

@model function as_turing_model(obs::TransformObservationModel, y_t, Y_t)
    transformed_Y_t = obs.transform(Y_t)
    y_t ~ to_submodel(as_turing_model(obs.model, y_t, transformed_Y_t), false)
    return y_t
end

# --- stacking ---------------------------------------------------------------

@doc raw"
Stack several observation models, each applied to a named component of the data.

Each inner model is wrapped in a [`PrefixObservationModel`](@ref) keyed by its
name, so the stacked variables stay distinct. The model is constructed either from
parallel vectors of models and names, or from a `NamedTuple` of models (the keys
supply the names). When sampled, each component model is applied to the matching
entry of the `y_t` / `Y_t` named tuples; a single expected-observation vector is
broadcast across all components.

# Arguments

  - `obs_model`: the [`StackObservationModels`](@ref) model.
  - `y_t`: a `NamedTuple` of observed series, one per stacked model.
  - `Y_t`: a `NamedTuple` of expected-observation series (or a single vector
    broadcast across the components).

# Examples
```@example StackObservationModels
using EpiAwarePrototype
obs = StackObservationModels((cases = PoissonError(), deaths = PoissonError()))
mdl = as_turing_model(obs, (cases = missing, deaths = missing), fill(10.0, 5))
rand(mdl)
```

## Fields

  - `models`: the vector of observation models (each prefix-wrapped by its name).
  - `model_names`: the names identifying each stacked model.
"
struct StackObservationModels{
    M <: AbstractVector, N <: AbstractVector{<:AbstractString}} <:
       AbstractEpiAwareModel
    "The vector of observation models (each prefix-wrapped by its name)."
    models::M
    "The names identifying each stacked model."
    model_names::N

    function StackObservationModels(
            models::M, model_names::N) where {
            M <: AbstractVector, N <: AbstractVector{<:AbstractString}}
        @assert length(models)==length(model_names) "The number of models and model names must be equal"
        prefix_models = [PrefixObservationModel(models[i], model_names[i])
                         for i in eachindex(models)]
        return new{typeof(prefix_models), N}(prefix_models, model_names)
    end
end

function StackObservationModels(models::NamedTuple)
    model_names = keys(models) .|> string |> collect
    return StackObservationModels(collect(values(models)), model_names)
end

@model function as_turing_model(
        obs_model::StackObservationModels, y_t::NamedTuple, Y_t::NamedTuple)
    @assert length(obs_model.models)==length(y_t) "The number of models must match the number of observed series"
    @assert obs_model.model_names==(keys(y_t) .|> string |> collect) "The model names must match the keys of the observed series"
    @assert keys(y_t)==keys(Y_t) "The keys of the observed and expected series must match"
    obs = Vector{Any}(undef, length(obs_model.models))
    for i in eachindex(obs_model.models)
        name = obs_model.model_names[i]
        obs_i ~ to_submodel(
            as_turing_model(
                obs_model.models[i], y_t[Symbol(name)], Y_t[Symbol(name)]),
            false)
        obs[i] = obs_i
    end
    return obs
end

@model function as_turing_model(
        obs_model::StackObservationModels, y_t::NamedTuple, Y_t::AbstractVector)
    tuple_Y_t = NamedTuple{keys(y_t)}(fill(Y_t, length(y_t)))
    obs ~ to_submodel(as_turing_model(obs_model, y_t, tuple_Y_t), false)
    return obs
end
