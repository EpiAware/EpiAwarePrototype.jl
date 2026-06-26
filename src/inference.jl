# Inference orchestration: a problem wrapper, inference methods (NUTS, multiple
# Pathfinder runs), the `apply_method` driver, and the observables container with
# a post-inference tidy helper. This layer is about *running* models, so unlike
# the model surface it keeps a small method type hierarchy (an optimisation
# pre-step feeding a sampler) — that is method dispatch, not model dispatch.

@doc raw"
Abstract supertype for inference / generative-modelling methods.
"
abstract type AbstractEpiMethod end

@doc raw"
Abstract supertype for optimisation-based methods (e.g. variational
initialisation) used as a pre-sampler step.
"
abstract type AbstractEpiOptMethod <: AbstractEpiMethod end

@doc raw"
Abstract supertype for sampling-based methods (e.g. NUTS).
"
abstract type AbstractEpiSamplingMethod <: AbstractEpiMethod end

@doc raw"
Combine a sequence of optimisation pre-steps with a sampler.

`apply_method(model, ::EpiMethod)` runs each `pre_sampler_steps` entry in turn,
threading the result into the next step and finally into the `sampler` (e.g.
using a [`ManyPathfinder`](@ref) result to initialise a [`NUTSampler`](@ref)).

## Fields

  - `pre_sampler_steps`: optimisation pre-steps (e.g. Pathfinder).
  - `sampler`: the sampler run last (e.g. NUTS).
"
@kwdef struct EpiMethod{O <: AbstractEpiOptMethod, S <: AbstractEpiSamplingMethod} <:
              AbstractEpiMethod
    "Optimisation pre-sampler steps."
    pre_sampler_steps::Vector{O}
    "The sampler run last."
    sampler::S
end

@doc raw"
A full epidemiological inference problem: a latent process, an infection process,
an observation model, and a time span.

`as_turing_model(problem, data)` assembles the corresponding [`EpiAwareModel`](@ref)
over `tspan` and conditions it on `data.y_t`.

# Arguments

  - `epiproblem`: the [`EpiProblem`](@ref).
  - `data`: a value with a `y_t` field holding the observations (or `missing`).

# Examples
```@example EpiProblem
using EpiAwarePrototype, Distributions
data = EpiData([0.2, 0.3, 0.5], exp)
problem = EpiProblem(
    epi_model = DirectInfections(; data = data, initialisation_prior = Normal()),
    latent_model = RandomWalk(),
    observation_model = PoissonError(),
    tspan = (1, 20))
rand(as_turing_model(problem, (; y_t = missing)))
```

## Fields

  - `epi_model`: the infection process model.
  - `latent_model`: the latent process model.
  - `observation_model`: the observation model.
  - `tspan`: the `(first, last)` time span of the series.
"
@kwdef struct EpiProblem{L <: AbstractEpiAwareModel, I <: AbstractEpiAwareModel,
    O <: AbstractEpiAwareModel}
    "The infection process model."
    epi_model::I
    "The latent process model."
    latent_model::L
    "The observation model."
    observation_model::O
    "The `(first, last)` time span of the series."
    tspan::Tuple{Int, Int}
end

@model function as_turing_model(epiproblem::EpiProblem, data)
    y_t = data.y_t
    time_steps = epiproblem.tspan[end] - epiproblem.tspan[1] + 1
    model = EpiAwareModel(
        epiproblem.latent_model, epiproblem.epi_model, epiproblem.observation_model)
    out ~ to_submodel(as_turing_model(model, y_t, time_steps), false)
    return out
end

@doc raw"
Container for the outputs of an inference run: the model, the data, the posterior
samples, and any generated quantities.

## Fields

  - `model`: the model that was sampled.
  - `data`: the data the model was conditioned on.
  - `samples`: the posterior samples (or optimiser result).
  - `generated`: generated quantities, or `missing` if not computed.
"
struct EpiAwareObservables{M, D, S, G}
    "The model that was sampled."
    model::M
    "The data the model was conditioned on."
    data::D
    "The posterior samples (or optimiser result)."
    samples::S
    "Generated quantities, or `missing`."
    generated::G
end

@doc raw"
Wrap a model, data, and inference solution into an [`EpiAwareObservables`](@ref).

# Arguments

  - `model`: the model that was sampled.
  - `data`: the data the model was conditioned on.
  - `solution`: the inference solution (samples or optimiser result).

# Examples
```@example generated_observables
using EpiAwarePrototype, Distributions
data = EpiData([0.2, 0.3, 0.5], exp)
m = as_turing_model(
    EpiAwareModel(RandomWalk(),
        DirectInfections(; data = data, initialisation_prior = Normal()),
        PoissonError()), missing, 10)
generated_observables(m, (; y_t = missing), rand(m))
```
"
function generated_observables(model, data, solution)
    return EpiAwareObservables(model, data, solution, missing)
end

@doc raw"
Condition a model by fixing some parameters and conditioning on others, then run
an inference `method`.

# Arguments

  - `epiproblem`: the [`EpiProblem`](@ref) (or a `DynamicPPL.Model`).
  - `method`: the inference method (a sampler or an [`EpiMethod`](@ref)).
  - `data`: the data to condition on (with a `y_t` field).

# Keyword Arguments

  - `fix_parameters`: a `NamedTuple` of parameters to fix.
  - `condition_parameters`: a `NamedTuple` of parameters to condition on.
  - `kwargs...`: forwarded to the inference method.

# Examples
```@example apply_method
using EpiAwarePrototype, Distributions
data = EpiData([0.2, 0.3, 0.5], exp)
problem = EpiProblem(
    epi_model = DirectInfections(; data = data, initialisation_prior = Normal()),
    latent_model = RandomWalk(),
    observation_model = PoissonError(),
    tspan = (1, 20))
y = rand(as_turing_model(problem, (; y_t = missing)))
nothing
```
"
function apply_method(epiproblem::EpiProblem, method::AbstractEpiMethod, data;
        fix_parameters::NamedTuple = NamedTuple(),
        condition_parameters::NamedTuple = NamedTuple(), kwargs...)
    model = as_turing_model(epiproblem, data)
    cond_model = condition_model(model, fix_parameters, condition_parameters)
    return apply_method(cond_model, method; kwargs...)
end

function apply_method(model::DynamicPPL.Model, method::EpiMethod, prev_result = nothing;
        kwargs...)
    for pre_sampler in method.pre_sampler_steps
        prev_result = _apply_method(model, pre_sampler, prev_result; kwargs...)
    end
    return _apply_method(model, method.sampler, prev_result; kwargs...)
end

# A bare method (sampler or optimiser) applied to a model goes straight to its
# `_apply_method` implementation.
function apply_method(model::DynamicPPL.Model, method::AbstractEpiMethod,
        prev_result = nothing; kwargs...)
    return _apply_method(model, method, prev_result; kwargs...)
end

@doc raw"
NUTS sampling method for a `DynamicPPL.Model`.

## Fields

  - `target_acceptance`: target acceptance rate.
  - `adtype`: automatic-differentiation backend.
  - `mcmc_parallel`: MCMC parallelisation strategy.
  - `nchains`: number of chains.
  - `max_depth`: NUTS tree-depth limit.
  - `Δ_max`: divergence threshold.
  - `init_ϵ`: initial step size (`0.0` lets NUTS find one).
  - `ndraws`: total draws.
  - `metricT`: HMC metric type.
  - `nadapts`: adaptation steps (`-1` uses the Turing default).
"
@kwdef struct NUTSampler{A <: ADTypes.AbstractADType,
    E <: AbstractMCMC.AbstractMCMCEnsemble, M} <: AbstractEpiSamplingMethod
    "Target acceptance rate."
    target_acceptance::Float64 = 0.8
    "Automatic-differentiation backend."
    adtype::A = AutoForwardDiff()
    "MCMC parallelisation strategy."
    mcmc_parallel::E = MCMCSerial()
    "Number of chains."
    nchains::Int = 1
    "NUTS tree-depth limit."
    max_depth::Int = 10
    "Divergence threshold."
    Δ_max::Float64 = 1000.0
    "Initial step size (`0.0` lets NUTS find one)."
    init_ϵ::Float64 = 0.0
    "Total draws."
    ndraws::Int
    "HMC metric type."
    metricT::M = DiagEuclideanMetric
    "Adaptation steps (`-1` uses the Turing default)."
    nadapts::Int = -1
end

function _apply_method(model::DynamicPPL.Model, method::NUTSampler, prev_result = nothing;
        kwargs...)
    return _apply_nuts(model, method, prev_result; kwargs...)
end

function _apply_nuts(model, method, prev_result; kwargs...)
    return sample(model,
        Turing.NUTS(method.target_acceptance; adtype = method.adtype,
            max_depth = method.max_depth, Δ_max = method.Δ_max,
            init_ϵ = method.init_ϵ, metricT = method.metricT),
        method.mcmc_parallel, method.ndraws ÷ method.nchains, method.nchains;
        nadapts = method.nadapts, kwargs...)
end

function _apply_nuts(model, method, prev_result::PathfinderResult; kwargs...)
    # A Pathfinder pre-step has run; thread its result through as the NUTS
    # initialisation. The mechanism by which earlier EpiAware seeded NUTS from a
    # Pathfinder draw (`init_params = eachrow(draws_transformed.value)`) is gone
    # in current Turing (`initial_params` now requires an `AbstractInitStrategy`,
    # not a vector) and Pathfinder (no `draws_transformed.value` array). The
    # `pathfinder` integration already initialises its own optimisation from the
    # model, so we run NUTS with the default strategy here; the Pathfinder result
    # remains available to the caller. Warm-starting NUTS from the draw will be
    # reinstated once the init-strategy API stabilises (tracked as a follow-up).
    return _apply_nuts(model, method, nothing; kwargs...)
end

@doc raw"
Direct sampling from a model's prior (no MCMC).

`apply_method(model, ::DirectSample)` samples the prior: with an integer
`n_samples` it draws that many times with `Turing.Prior()` (returning a chain),
and with `nothing` it draws once with `rand` (returning a `NamedTuple`).

## Fields

  - `n_samples`: number of prior draws, or `nothing` for a single `rand` draw.
"
@kwdef struct DirectSample <: AbstractEpiSamplingMethod
    "Number of prior draws, or `nothing` for a single `rand` draw."
    n_samples::Union{Int, Nothing} = nothing
end

function _apply_method(model::DynamicPPL.Model, method::DirectSample,
        prev_result = nothing; kwargs...)
    return _apply_direct_sample(model, method, method.n_samples; kwargs...)
end

function _apply_direct_sample(model, method, n_samples::Int; kwargs...)
    sample(
        model, Turing.Prior(), n_samples; kwargs...)
end
_apply_direct_sample(model, method, ::Nothing; kwargs...) = rand(model)

@doc raw"
Reshape an `MCMCChains.Chains` object into a `(draws × chains)` array of
per-sample `NamedTuple`s.

# Arguments

  - `chn`: the `Chains` object.

# Examples
```@example get_param_array
using EpiAwarePrototype
nothing
```
"
function get_param_array(chn::Chains)
    return rowtable(chn) |> x -> reshape(x, size(chn, 1), size(chn, 3))
end

@doc raw"
Variational pre-sampler that runs Pathfinder several times and keeps the best run.

## Fields

  - `ndraws`: draws per Pathfinder run.
  - `nruns`: number of Pathfinder runs.
  - `maxiters`: optimiser iterations per run.
  - `max_tries`: extra tries if all runs fail.
"
@kwdef struct ManyPathfinder <: AbstractEpiOptMethod
    "Draws per Pathfinder run."
    ndraws::Int = 10
    "Number of Pathfinder runs."
    nruns::Int = 4
    "Optimiser iterations per run."
    maxiters::Int = 100
    "Extra tries if all runs fail."
    max_tries::Int = 100
end

function _apply_method(model::DynamicPPL.Model, method::ManyPathfinder,
        prev_result = nothing; kwargs...)
    return _apply_pathfinder(model, method, prev_result; kwargs...)
end

function _apply_pathfinder(model, method, prev_result; kwargs...)
    return manypathfinder(model, method.ndraws; nruns = method.nruns,
        maxiters = method.maxiters, kwargs...)
end

function _apply_pathfinder(model, method, prev_result::Vector{<:Real}; kwargs...)
    return manypathfinder(model, method.ndraws; init = prev_result,
        nruns = method.nruns, maxiters = method.maxiters, kwargs...)
end

@doc raw"
Run Pathfinder several times and return the run with the largest ELBO estimate.

# Arguments

  - `mdl`: the `DynamicPPL.Model` to fit.
  - `ndraws`: draws per Pathfinder run.

# Keyword Arguments

  - `nruns`: number of Pathfinder runs (default `4`).
  - `maxiters`: optimiser iterations per run (default `50`).
  - `max_tries`: extra tries if all runs fail (default `100`).
  - `kwargs...`: forwarded to `pathfinder`.

# Examples
```@example manypathfinder
using EpiAwarePrototype, Distributions
data = EpiData([0.2, 0.3, 0.5], exp)
m = as_turing_model(
    EpiAwareModel(RandomWalk(),
        DirectInfections(; data = data, initialisation_prior = Normal()),
        PoissonError()), fill(10, 10), 10)
nothing
```
"
function manypathfinder(mdl::DynamicPPL.Model, ndraws; nruns = 4, maxiters = 50,
        max_tries = 100, kwargs...)
    return _run_manypathfinder(mdl; nruns, ndraws, maxiters, kwargs...) |>
           pfs -> _continue_manypathfinder!(pfs, mdl; max_tries, nruns, kwargs...) |>
                  pfs -> _get_best_elbo_pathfinder(pfs)
end

function _run_manypathfinder(mdl::DynamicPPL.Model; nruns, kwargs...)
    @info "Running pathfinder $nruns times"
    pfs = Vector{Union{PathfinderResult, Symbol}}(undef, nruns)
    Threads.@threads for i in 1:nruns
        try
            pfs[i] = pathfinder(mdl; kwargs...)
        catch
            pfs[i] = :fail
        end
    end
    return pfs
end

function _continue_manypathfinder!(pfs, mdl::DynamicPPL.Model; max_tries, nruns,
        kwargs...)
    tryiter = 1
    if all(pfs .== :fail)
        @warn "All initial pathfinder runs failed, trying again for $max_tries tries."
    end
    while all(pfs .== :fail) && tryiter <= max_tries
        new_pf = try
            pathfinder(mdl; kwargs...)
        catch
            :fail
        end
        pfs = vcat(pfs, new_pf)
        tryiter += 1
    end
    if all(pfs .== :fail)
        throw(ErrorException("All pathfinder runs failed after $max_tries tries."))
    end
    return pfs
end

function _get_best_elbo_pathfinder(pfs)
    elbos = map(pfs) do pf_res
        pf_res == :fail ? -Inf : pf_res.elbo_estimates[end].value
    end
    _, choice_of_pf = findmax(elbos)
    return pfs[choice_of_pf]
end

@doc raw"
Convert an `MCMCChains.Chains` object to a tidy `DataFrame` (one row per draw,
with `draw`, `chain`, and `iteration` columns).

# Arguments

  - `chn`: the `Chains` object to convert.

# Examples
```@example spread_draws
using EpiAwarePrototype
nothing
```
"
function spread_draws(chn::Chains)
    df = DataFrame(chn)
    # `DataFrame(::Chains)` emits the bookkeeping columns as `.iteration` and
    # `.chain` (current MCMCChains); normalise them and add a sequential `draw`
    # index in tidybayes style. Older MCMCChains used undotted names, so accept
    # either.
    for (dotted, plain) in ((".iteration", "iteration"), (".chain", "chain"))
        if dotted in names(df)
            @rename!(df, $(plain)=$(dotted))
        end
    end
    df = hcat(DataFrame(draw = 1:size(df, 1)), df)
    return df
end
