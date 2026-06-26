# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# AD-fixture registry implementing the EpiAwarePackageTools `ADRegistry`
# contract. The scenarios are REAL differentiable log-densities from the
# package: the (linked) log-joint of representative latent processes and of
# composed `EpiAwareModel`s conditioned on simulated data — the gradients an AD
# backend must get right for NUTS to work. Each scenario carries a ForwardDiff
# reference gradient. The shared harness (driven from `test/ad/setup.jl`)
# consumes this registry.
module ADFixtures

using ADTypes: AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface
import DifferentiationInterfaceTest as DIT
import ForwardDiff
using EpiAwarePrototype
using Distributions
using Random: Random, MersenneTwister
using DynamicPPL: DynamicPPL, LogDensityFunction, VarInfo, link, getlogjoint
import LogDensityProblems as LDP

export scenarios, backends, broken_scenario_names,
       backend_broken_scenarios, backend_skip_scenarios

# Turn a DynamicPPL model into a real differentiable scalar log-density.
#
# We link the model's `VarInfo` so every constrained variable (truncated damping
# priors, positive standard deviations, simplex-free reals, ...) maps to an
# unconstrained real coordinate. The returned `f(θ)` is then the log-joint
# (including the linking log-Jacobian) at the flat unconstrained vector `θ`, and
# is finite and smooth over all of ``ℝ^d`` — exactly the target a gradient-based
# sampler differentiates. Returns `(f, θ0, dim)`.
function _logdensity(model; seed::Int = 1)
    vi = link(VarInfo(model), model)
    ldf = LogDensityFunction(model, getlogjoint, vi)
    dim = LDP.dimension(ldf)
    f = θ -> LDP.logdensity(ldf, θ)
    θ0 = 0.3 .* randn(MersenneTwister(seed), dim)
    return f, θ0, dim
end

# A representative generation interval shared by the infection-model scenarios.
const _GEN_INT = [0.2, 0.3, 0.5]

# Build the registry's models once. Conditioned (posterior) scenarios use data
# simulated from the prior with a fixed seed so the target is deterministic.
function _models()
    data = EpiData(_GEN_INT, exp)
    n = 12

    rw = as_turing_model(RandomWalk(), n)
    ar = as_turing_model(AR(), n)
    arima = as_turing_model(
        DiffLatentModel(; model = AR(), init_priors = [Normal(), Normal()]), n)

    direct = EpiAwareModel(RandomWalk(),
        DirectInfections(; data = data, initialisation_prior = Normal()),
        PoissonError())
    renewal = EpiAwareModel(RandomWalk(),
        Renewal(data; initialisation_prior = Normal()),
        NegativeBinomialError())

    y_direct = as_turing_model(direct, missing, n)().generated_y_t
    y_renewal = as_turing_model(renewal, missing, n)().generated_y_t

    return [
        ("RandomWalk latent logjoint", rw),
        ("AR latent logjoint", ar),
        ("ARIMA latent logjoint", arima),
        ("DirectInfections+Poisson posterior",
            as_turing_model(direct, y_direct, n)),
        ("Renewal+NegativeBinomial posterior",
            as_turing_model(renewal, y_renewal, n))
    ]
end

@doc """
    scenarios(; with_reference = false, category = :marginal)

The AD gradient scenarios — each a `DIT.Scenario{:gradient, :out}` over a real
package log-density (a latent process prior log-joint, or a composed
`EpiAwareModel` posterior conditioned on simulated data). When
`with_reference = true` each scenario carries its ForwardDiff reference gradient
in `res1`. `category` is accepted for the harness's group selector; all
scenarios are in the single `:marginal` group here.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    out = DIT.Scenario{:gradient, :out}[]
    for (i, (name, model)) in enumerate(_models())
        f, θ0, _ = _logdensity(model; seed = i)
        ref = with_reference ?
              DifferentiationInterface.gradient(f, AutoForwardDiff(), θ0) :
              nothing
        push!(out,
            DIT.Scenario{:gradient, :out}(f, θ0; name = name, res1 = ref))
    end
    return out
end

@doc """
    backends()

The AD backends exercised against the scenarios, as `(; name, backend)` named
tuples: ForwardDiff (the reference), ReverseDiff (tape), Mooncake, and Enzyme
reverse. Per-backend brokenness is recorded honestly in
[`backend_broken_scenarios`](@ref) / [`broken_scenario_names`](@ref) rather than
by trimming this list.
"""
function backends()
    return [
        (name = "ForwardDiff", backend = _forwarddiff()),
        (name = "ReverseDiff (tape)", backend = _reversediff()),
        (name = "Mooncake reverse", backend = _mooncake()),
        (name = "Enzyme reverse", backend = _enzyme())
    ]
end

# Backend constructors are written so that loading a backend package is only
# required when that backend is actually requested (the AD env loads them all,
# but this keeps the registry importable without every backend present).
_forwarddiff() = AutoForwardDiff()
function _reversediff()
    ADTypes = Base.require(Base.PkgId(
        Base.UUID("47edcb42-4c32-4615-8424-f2b9edc5f35b"), "ADTypes"))
    return ADTypes.AutoReverseDiff(; compile = false)
end
function _mooncake()
    ADTypes = Base.require(Base.PkgId(
        Base.UUID("47edcb42-4c32-4615-8424-f2b9edc5f35b"), "ADTypes"))
    return ADTypes.AutoMooncake(; config = nothing)
end
function _enzyme()
    ADTypes = Base.require(Base.PkgId(
        Base.UUID("47edcb42-4c32-4615-8424-f2b9edc5f35b"), "ADTypes"))
    Enzyme = Base.require(Base.PkgId(
        Base.UUID("7da242da-08ed-463a-9acd-ee780be4f1d9"), "Enzyme"))
    # `function_annotation = Enzyme.Const`: the log-density closures carry no
    # derivative data, and without this Enzyme raises `EnzymeMutabilityException`
    # ("argument cannot be proven readonly") on every DynamicPPL log-density.
    # With it, three of the five scenarios differentiate correctly; the AR-based
    # two remain genuinely broken (see `backend_broken_scenarios`).
    return ADTypes.AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Const)
end

"Scenario names broken on every backend (none — all are real, FD-differentiable)."
broken_scenario_names() = String[]

@doc """
    backend_broken_scenarios()

Per-backend broken scenario names (`Dict{String, Set{String}}`), populated
HONESTLY from the actual `test/ad` run rather than by silencing.

Result matrix (5 scenarios × 4 backends), Julia 1.12:

| scenario                            | ForwardDiff | ReverseDiff | Mooncake | Enzyme |
|-------------------------------------|:-----------:|:-----------:|:--------:|:------:|
| RandomWalk latent logjoint          |      ✓      |      ✓      |    ✓    |   ✓   |
| AR latent logjoint                  |      ✓      |      ✓      |    ✓    |   ✗   |
| ARIMA latent logjoint               |      ✓      |      ✓      |    ✓    |   ✗   |
| DirectInfections+Poisson posterior  |      ✓      |      ✓      |    ✓    |   ✓   |
| Renewal+NegativeBinomial posterior  |      ✓      |      ✓      |    ✓    |   ✓   |

ForwardDiff (the reference), ReverseDiff, and Mooncake differentiate every
scenario correctly. Enzyme works on three of the five once configured with
`function_annotation = Enzyme.Const` (see [`backends`](@ref)), but the two
AR-based latent log-densities raise `IllegalTypeAnalysisException` inside the
`accumulate_scan(ARStep(damp_AR), ...)` / `LinearAlgebra.dot` recursion — a real
Enzyme type-analysis limitation, not a defect in the package (the same models
sample fine under NUTS with ForwardDiff). They are recorded as `@test_broken`
for Enzyme below rather than hidden.
"""
function backend_broken_scenarios()
    return Dict{String, Set{String}}(
        "Enzyme reverse" => Set([
        "AR latent logjoint", "ARIMA latent logjoint"]))
end

"Per-backend scenario names too unstable to even run (segfault/hang)."
backend_skip_scenarios() = Dict{String, Set{String}}()

end # module ADFixtures
