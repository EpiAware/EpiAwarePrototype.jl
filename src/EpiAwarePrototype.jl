@doc raw"
A **prototype** for composable probabilistic infectious disease modelling in
Julia.

`EpiAwarePrototype` builds epidemiological models from small, reusable
components — latent processes, infection processes, and observation models —
each turned into a `Turing`/`DynamicPPL` model by the single generic constructor
[`as_turing_model`](@ref). Components compose by sampling one another as
submodels, so a full model is assembled rather than hand-written.

This package is **ported and adapted** from the open-source, Apache-2.0 licensed
`EpiAware` package; see the `NOTICE` file for attribution. It is exploratory and
clearly labelled as a prototype.

# Examples
```@example
using EpiAwarePrototype, Distributions
data = EpiData([0.2, 0.3, 0.5], exp)
model = EpiAwareModel(RandomWalk(),
    DirectInfections(; data = data, initialisation_prior = Normal()),
    PoissonError())
rand(as_turing_model(model, missing, 20))
```
"
module EpiAwarePrototype

using Reexport: @reexport
@reexport using Distributions
@reexport using Turing

using DynamicPPL: DynamicPPL, @model, to_submodel, fix, condition, prefix
using Turing: Turing, filldist, arraydist, sample, NUTS, MCMCSerial
# `sampler` is imported only to satisfy ExplicitImports: the `EpiMethod` field /
# keyword named `sampler` (kept for parity) collides with the reexported
# `Distributions.sampler`, which the analysis otherwise reports as implicit.
using Turing: sampler
using LinearAlgebra: dot
using LogExpFunctions: softmax, xexpy, log1pexp
using OrdinaryDiffEq: ODEProblem, ODEFunction, solve, remake, AutoVern7, Rodas5P
using QuadGK: quadgk
using Random: AbstractRNG, randexp

# Inference-layer dependencies.
using ADTypes: ADTypes, AutoForwardDiff
using AbstractMCMC: AbstractMCMC
using AdvancedHMC: DiagEuclideanMetric
using MCMCChains: Chains
using Pathfinder: pathfinder, PathfinderResult
using DataFramesMeta: DataFrame, @rename!
using Tables: rowtable

# Names used (and, for many, extended) by the prototype. Imported explicitly so
# the package surface stays analysable by ExplicitImports even though the whole
# of Distributions/Turing is reexported for users above.
using Distributions: Distributions, Distribution, Sampleable,
                     ContinuousUnivariateDistribution, ContinuousDistribution,
                     Normal, Poisson, NegativeBinomial, Gamma, truncated,
                     cdf, ccdf, logcdf, logccdf, invlogcdf, pdf, logpdf, quantile,
                     params, mean, var, std, mode, skewness, kurtosis, succprob,
                     failprob
using Statistics: Statistics

# --- core architecture ---
export AbstractEpiAwareModel, as_turing_model

# --- utilities and distributions ---
export accumulate_scan, get_state, HalfNormal, SafePoisson, SafeNegativeBinomial,
       NegativeBinomialMeanClust, censored_pmf, censored_cdf, ∫F, condition_model

# --- latent models ---
export IID, HierarchicalNormal, RandomWalk, AR, MA, Intercept, FixedIntercept,
       Null, DiffLatentModel

# --- latent modifiers / manipulators / combinations / broadcasting ---
export TransformLatentModel, PrefixLatentModel, RecordExpectedLatent,
       CombineLatentModels, ConcatLatentModels, BroadcastLatentModel,
       RepeatEach, RepeatBlock, broadcast_rule, broadcast_n, broadcast_dayofweek,
       broadcast_weekly, equal_dimensions, arma, arima

# --- infection models ---
export EpiData, DirectInfections, ExpGrowthRate, Renewal,
       R_to_r, r_to_R, expected_Rt

# --- ODE compartmental models ---
export SIRParams, SEIRParams, ODEProcess

# --- observation models ---
export PoissonError, NegativeBinomialError, LatentDelay,
       observation_error, generate_observation_error_priors

# --- observation modifiers / manipulators ---
export Ascertainment, ascertainment_dayofweek, Aggregate, PrefixObservationModel,
       RecordExpectedObs, TransformObservationModel, StackObservationModels

# --- composition ---
export EpiAwareModel

# --- inference orchestration ---
export EpiProblem, EpiMethod, NUTSampler, ManyPathfinder, DirectSample,
       manypathfinder, apply_method, EpiAwareObservables, generated_observables,
       spread_draws, get_param_array

include("base.jl")
include("utils.jl")
include("latent.jl")
include("latent_extra.jl")
include("infections.jl")
include("infections_extra.jl")
include("ode.jl")
include("observations.jl")
include("observations_extra.jl")
include("compose.jl")
include("inference.jl")

end
