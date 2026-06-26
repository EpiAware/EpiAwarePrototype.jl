# Compartmental ODE models (SIR, SEIR) and the `ODEProcess` infection model that
# solves them. The parameter structs (`SIRParams`, `SEIRParams`) play the role of
# a latent model: their `as_turing_model` samples the ODE parameters and returns
# the `(u0, p)` initial-condition/parameter tuple. `ODEProcess` wraps a parameter
# struct with a solver and a `sol2infs` link to produce latent infections.

# --- SIR --------------------------------------------------------------------

# Vector field of the density/per-capita SIR model.
function _sir_vf(du, u, p, t)
    S, Iv, R = u
    β, γ = p
    du[1] = -β * S * Iv
    du[2] = β * S * Iv - γ * Iv
    du[3] = γ * Iv
    return nothing
end

# Jacobian of the SIR vector field (speeds up stiff solves).
function _sir_jac(J, u, p, t)
    S, Iv, R = u
    β, γ = p
    J[1, 1] = -β * Iv
    J[1, 2] = -β * S
    J[2, 1] = β * Iv
    J[2, 2] = β * S - γ
    J[3, 2] = γ
    return nothing
end

const _sir_function = ODEFunction(_sir_vf; jac = _sir_jac)

@doc raw"
SIR compartmental model parameters and priors, usable as the latent component of
an [`ODEProcess`](@ref).

```math
\frac{dS}{dt} = -\beta S I, \quad
\frac{dI}{dt} = \beta S I - \gamma I, \quad
\frac{dR}{dt} = \gamma I
```

# Arguments

  - `params`: the [`SIRParams`](@ref) struct.
  - `n`: unused size argument (the ODE dimension is fixed); accepted for the
    common `as_turing_model` signature.

# Keyword Arguments

  - `tspan`: the ODE solution time span.
  - `infectiousness`: prior for ``\beta``.
  - `recovery_rate`: prior for ``\gamma``.
  - `initial_prop_infected`: prior for the initial infected proportion.

# Examples
```@example SIRParams
using EpiAwarePrototype, OrdinaryDiffEq, Distributions
sirparams = SIRParams(
    tspan = (0.0, 30.0),
    infectiousness = LogNormal(log(0.3), 0.05),
    recovery_rate = LogNormal(log(0.1), 0.05),
    initial_prop_infected = Beta(1, 99))
rand(as_turing_model(sirparams, nothing))
```

## Fields

  - `prob`: the `ODEProblem` instance for the SIR model.
  - `infectiousness`: prior for ``\beta``.
  - `recovery_rate`: prior for ``\gamma``.
  - `initial_prop_infected`: prior for the initial infected proportion.
"
struct SIRParams{P <: ODEProblem, D <: Sampleable, E <: Sampleable, F <: Sampleable} <:
       AbstractEpiAwareModel
    "The `ODEProblem` instance for the SIR model."
    prob::P
    "Prior for the infectiousness parameter."
    infectiousness::D
    "Prior for the recovery rate parameter."
    recovery_rate::E
    "Prior for the initial infected proportion."
    initial_prop_infected::F
end

function SIRParams(; tspan, infectiousness::Distribution, recovery_rate::Distribution,
        initial_prop_infected::Distribution)
    sir_prob = ODEProblem(_sir_function, [0.99, 0.01, 0.0], tspan)
    return SIRParams{typeof(sir_prob), typeof(infectiousness),
        typeof(recovery_rate), typeof(initial_prop_infected)}(
        sir_prob, infectiousness, recovery_rate, initial_prop_infected)
end

@model function as_turing_model(params::SIRParams, n)
    β ~ params.infectiousness
    γ ~ params.recovery_rate
    I₀ ~ params.initial_prop_infected
    u0 = [1.0 - I₀, I₀, 0.0]
    p = [β, γ]
    return (u0, p)
end

# --- SEIR -------------------------------------------------------------------

function _seir_vf(du, u, p, t)
    S, E, Iv, R = u
    β, α, γ = p
    du[1] = -β * S * Iv
    du[2] = β * S * Iv - α * E
    du[3] = α * E - γ * Iv
    du[4] = γ * Iv
    return nothing
end

function _seir_jac(J, u, p, t)
    S, E, Iv, R = u
    β, α, γ = p
    J[1, 1] = -β * Iv
    J[1, 3] = -β * S
    J[2, 1] = β * Iv
    J[2, 2] = -α
    J[2, 3] = β * S
    J[3, 2] = α
    J[3, 3] = -γ
    J[4, 3] = γ
    return nothing
end

const _seir_function = ODEFunction(_seir_vf; jac = _seir_jac)

@doc raw"
SEIR compartmental model parameters and priors, usable as the latent component of
an [`ODEProcess`](@ref).

```math
\frac{dS}{dt} = -\beta S I, \quad
\frac{dE}{dt} = \beta S I - \alpha E, \quad
\frac{dI}{dt} = \alpha E - \gamma I, \quad
\frac{dR}{dt} = \gamma I
```

The sampled initial infected proportion is split between the exposed and
infectious compartments using the constant-incidence equilibrium proportions
``\gamma/(\alpha+\gamma)`` and ``\alpha/(\alpha+\gamma)``.

# Arguments

  - `params`: the [`SEIRParams`](@ref) struct.
  - `n`: unused size argument; accepted for the common `as_turing_model`
    signature.

# Keyword Arguments

  - `tspan`: the ODE solution time span.
  - `infectiousness`: prior for ``\beta``.
  - `incubation_rate`: prior for ``\alpha``.
  - `recovery_rate`: prior for ``\gamma``.
  - `initial_prop_infected`: prior for the initial infected proportion.

# Examples
```@example SEIRParams
using EpiAwarePrototype, OrdinaryDiffEq, Distributions
seirparams = SEIRParams(
    tspan = (0.0, 30.0),
    infectiousness = LogNormal(log(0.3), 0.05),
    incubation_rate = LogNormal(log(0.1), 0.05),
    recovery_rate = LogNormal(log(0.1), 0.05),
    initial_prop_infected = Beta(1, 99))
rand(as_turing_model(seirparams, nothing))
```

## Fields

  - `prob`: the `ODEProblem` instance for the SEIR model.
  - `infectiousness`: prior for ``\beta``.
  - `incubation_rate`: prior for ``\alpha``.
  - `recovery_rate`: prior for ``\gamma``.
  - `initial_prop_infected`: prior for the initial infected proportion.
"
struct SEIRParams{P <: ODEProblem, D <: Sampleable, E <: Sampleable,
    F <: Sampleable, G <: Sampleable} <: AbstractEpiAwareModel
    "The `ODEProblem` instance for the SEIR model."
    prob::P
    "Prior for the infectiousness parameter."
    infectiousness::D
    "Prior for the incubation rate parameter."
    incubation_rate::E
    "Prior for the recovery rate parameter."
    recovery_rate::F
    "Prior for the initial infected proportion."
    initial_prop_infected::G
end

function SEIRParams(; tspan, infectiousness::Distribution, incubation_rate::Distribution,
        recovery_rate::Distribution, initial_prop_infected::Distribution)
    seir_prob = ODEProblem(_seir_function, [0.99, 0.05, 0.05, 0.0], tspan)
    return SEIRParams{typeof(seir_prob), typeof(infectiousness),
        typeof(incubation_rate), typeof(recovery_rate),
        typeof(initial_prop_infected)}(seir_prob, infectiousness,
        incubation_rate, recovery_rate, initial_prop_infected)
end

@model function as_turing_model(params::SEIRParams, n)
    β ~ params.infectiousness
    α ~ params.incubation_rate
    γ ~ params.recovery_rate
    initial_infs ~ params.initial_prop_infected
    u0 = [1.0 - initial_infs, initial_infs * γ / (α + γ),
        initial_infs * α / (α + γ), 0.0]
    p = [β, α, γ]
    return (u0, p)
end

# --- ODEProcess -------------------------------------------------------------

@doc raw"
An infection process defined by solving an ODE.

`ODEProcess` combines a parameter struct (`params`, e.g. [`SIRParams`](@ref) or
[`SEIRParams`](@ref), whose `as_turing_model` samples `(u0, p)`) with a `solver`,
extra `solver_options`, and a `sol2infs` link mapping the ODE solution to a
latent-infection series. Its `as_turing_model` samples the parameters, solves the
ODE, and returns the mapped infections.

# Arguments

  - `epi_model`: the [`ODEProcess`](@ref).
  - `Z_t`: an optional latent path (may be `nothing` for self-contained ODE
    models); only its length is used.

# Examples
```@example ODEProcess
using EpiAwarePrototype, OrdinaryDiffEq, Distributions, LogExpFunctions
sirparams = SIRParams(
    tspan = (0.0, 100.0),
    infectiousness = LogNormal(log(0.3), 0.05),
    recovery_rate = LogNormal(log(0.1), 0.05),
    initial_prop_infected = Beta(1, 99))
N = 1000.0
sir_process = ODEProcess(
    params = sirparams,
    sol2infs = sol -> softplus.(N .* sol[2, :]),
    solver_options = Dict(:saveat => 1.0))
as_turing_model(sir_process, nothing)()
```

## Fields

  - `params`: the ODE parameter model (an `AbstractEpiAwareModel`).
  - `solver`: the ODE solver (default `AutoVern7(Rodas5P())`).
  - `sol2infs`: link mapping the ODE solution to an infection series.
  - `solver_options`: extra options passed to `solve` (a `Dict` or `NamedTuple`).
"
@kwdef struct ODEProcess{P <: AbstractEpiAwareModel, S, F <: Function,
    D <: Union{Dict, NamedTuple}} <: AbstractEpiAwareModel
    "The ODE parameter model."
    params::P
    "The ODE solver."
    solver::S = AutoVern7(Rodas5P())
    "Link mapping the ODE solution to an infection series."
    sol2infs::F
    "Extra options passed to `solve`."
    solver_options::D = Dict(:saveat => 1.0)
end

# Sample the ODE parameters and solve, returning the solution object.
@model function _generate_ode_solution(epi_model::ODEProcess, n)
    prob = epi_model.params.prob
    solver = epi_model.solver
    solver_options = epi_model.solver_options
    params ~ to_submodel(as_turing_model(epi_model.params, n), false)
    u0, p = params
    _prob = remake(prob; u0 = u0, p = p)
    sol = solve(_prob, solver; solver_options...)
    return sol
end

@model function as_turing_model(epi_model::ODEProcess, Z_t)
    n = isnothing(Z_t) ? 0 : size(Z_t, 1)
    sol ~ to_submodel(_generate_ode_solution(epi_model, n), false)
    return epi_model.sol2infs(sol)
end
