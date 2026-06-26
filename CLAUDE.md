# EpiAwarePrototype.jl

A **prototype** for composable probabilistic infectious disease modelling in Julia.
Goal: reach a state external collaborators can install and test. Treat everything
here as exploratory and clearly labelled as a prototype.

> **Read this file before starting any work.** It is the standing brief for every
> agent and contributor on this repo. Keep the "Current status" section at the
> bottom up to date as work lands.

## Provenance, attribution & licensing (IMPORTANT)

- The modelling code in this package is **ported and adapted** from the
  open-source, **Apache-2.0** licensed `EpiAware` package developed at CDC:
  - Upstream: https://github.com/CDCgov/Rt-without-renewal (`EpiAware/`)
  - Fork ported from: https://github.com/seabbs/Rt-without-renewal
- Because we incorporate Apache-2.0 code, **this package is licensed Apache-2.0**
  for compatibility. We must:
  - Keep the Apache-2.0 `LICENSE`.
  - Ship a `NOTICE` file attributing the upstream authors and stating that this
    is a modified/derived work, with a link back to the upstream repo.
  - Add a short **"Adapted from"** disclaimer in the `README` and docs index.
  - State that significant changes have been made (renamed, re-architected
    around `as_turing_model`, upgraded to the latest Turing).
- Attribute **only** the open Apache-2.0 code above. Do **not** reference any
  non-public repositories or unpublished material in package-facing files, commit
  messages, issues, or docs. All package documentation must be original to this
  package.

## Tooling & template

- This package is scaffolded from **`EpiAwarePackageTools.jl`** (the
  `EpiAwareTestUtils` kit): https://github.com/EpiAware/EpiAwarePackageTools.jl
- Use its `scaffold(pkgdir)` / `update(pkgdir)` and its test helpers
  (`test_aqua`, `test_jet`, `test_doctest`, `test_formatting`, AD harness, etc.)
  **instead of re-inventing** test/CI/dev scaffolding.
- **Adopt the template; adapt our ported code to fit it** — not the other way
  round.
- If the template or `scaffold`/`update` does not work, is missing something, or
  forces an awkward workaround, **file an issue against
  `EpiAware/EpiAwarePackageTools.jl`** describing the gap, rather than silently
  patching around it locally. Track such issues in the status section below.

## Architecture directive: `as_turing_model`

Replace the upstream abstract **type hierarchy** + per-concept generate
functions (`generate_latent`, `generate_observations`, `generate_latent_infs`,
`generate_epiaware`, dispatching on `AbstractLatentModel` /
`AbstractObservationModel` / `AbstractEpiModel`) with a **single generic
constructor**:

```julia
as_turing_model(model, args...; kwargs...)  # returns a DynamicPPL.Model
```

- Every model struct implements one `@model function as_turing_model(m::MyModel, ...)`.
- Compose via submodels of `as_turing_model(component, ...)`.
- Keep backend-agnostic pieces backend-agnostic (`accumulate_scan`,
  `AbstractAccumulationStep` step structs, distribution/utility helpers).
- Collapse the deep abstract hierarchy. A single light supertype (e.g.
  `AbstractEpiAwareModel`) for shared behaviour/printing is fine; the deep
  `AbstractTuring*` tree is not needed.

## Turing / DynamicPPL: target the latest

- Build against the **latest** released `Turing.jl` / `DynamicPPL.jl`.
- The `@submodel` macro is **removed**. Use the tilde + `to_submodel` form:

  ```julia
  # old:  @submodel ϵ_t = generate_latent(m.ϵ_t, n - p)
  # new:  ϵ_t ~ to_submodel(as_turing_model(m.ϵ_t, n - p), false)
  ```

  **Prefix off is the standard here.** The current DynamicPPL default for
  `to_submodel` is **prefix = true** (it prefixes the submodel's variables with
  the left-hand name), which differs from upstream's flat `@submodel` behaviour.
  Pass `false` as the standard on *every* submodel conversion to preserve the
  existing variable names/behaviour. The only exceptions are the components that
  upstream deliberately prefixed (`PrefixLatentModel`, `PrefixObservationModel`,
  `StackObservationModels`, the old `prefix_submodel` call sites) — implement
  their prefixing explicitly via `to_submodel`'s prefix argument.

## Naming

- Package name and top-level module: **`EpiAwarePrototype`** (fresh UUID in
  `Project.toml`). Update all references, docstrings, doctests, and CI badges.

## Docs

- Make it clear throughout that this is a **prototype for composable infectious
  disease modelling**.
- Document the composable-modelling design (the component DSL idea + the
  Turing.jl backend / `as_turing_model` API) with **original, package-specific
  documentation** written for this package.
- **Declutter:** remove stub/placeholder pages for features we are not shipping
  in the prototype. Keep a focused, honest set of pages (getting started, the
  composable design, a worked example, API reference).

## Workflow

- **Until a working port exists** (package loads + a representative end-to-end
  model samples + core tests pass): commit **directly to `main`** in the local
  clone. No PR overhead yet.
- **Once the port works:** add **branch protection** to `main`, then switch to a
  **review-PR workflow** for all further changes.

## Build strategy

**Port the COMPLETE package.** Every upstream source file, module, model, helper,
manipulator/modifier, inference method, the problem/method glue, the tests, and the
docs — all of it, ported and adapted to the new `as_turing_model` API on the latest
Turing. **Nothing stubbed, nothing deferred, no functionality dropped.** Full feature
parity with upstream `EpiAware`, just re-architected and renamed.

The scaffold-from-EpiAwarePackageTools, Apache-2.0 licence + NOTICE/attribution,
originality, commit-to-`main`, and issue-logging rules are *how* the port is done
— they are not a reduction of scope.

Sequence the work so `main` stays loadable (port in dependency order, commit in
logical chunks). The deliverable is the **complete, working package**: it loads, every
ported model can be constructed and sampled, end-to-end composed models run
(rand/fix/condition/NUTS), and the full EpiAwareTestUtils test suite passes. Only
switch to the review-PR workflow once that complete version works.

### Porting order — submodule by submodule, starting with the core

Port and stabilise **one submodule at a time**, in dependency order, getting each to
load + its tests pass before moving on:

1. `EpiAwareBase` (core: the single supertype, the `as_turing_model` generic, glue)
2. `EpiAwareUtils` (accumulate_scan, distributions/helpers, submodel handling)
3. `EpiLatentModels`
4. `EpiInfModels`
5. `EpiObsModels`
6. `EpiInference`
7. `EpiProblem` / `EpiMethod` glue

### Dependency: use CensoredDistributions.jl for censoring

Upstream rolls its own **double interval censoring** (`censored_pmf` / `censored_cdf`
in `EpiAwareUtils/censored_pmf.jl`). **Do not port that bespoke code.** Instead depend
on **CensoredDistributions.jl** (the EpiAware org package; local at
`~/code/EpiAware/CensoredDistributions.jl`, exports `double_interval_censored`,
`interval_censored`, `primary_censored`, …) and use it to produce the discretised
PMFs. Affected call sites to migrate:
- `EpiInfModels/EpiData.jl` — generation-interval discretisation (`gen_int = censored_pmf(...)`).
- `EpiObsModels/modifiers/LatentDelay.jl` — delay PMF (`pmf = censored_pmf(...)`).

Read CensoredDistributions.jl's API/docs to find the right call that yields the
right-truncated discretised PMF vector these sites need. Drop `censored_pmf.jl`.

> **UUID caution:** CensoredDistributions.jl currently shares the *same* UUID as the
> upstream EpiAware package (`b2eeebe4-5992-4301-9193-7ebc9f62c855`). EpiAwarePrototype
> MUST take a fresh, distinct UUID, and the env must resolve CensoredDistributions by
> its registered/repo UUID — watch for a clash.

## Current status

- [x] Repo scaffolded from EpiAwarePackageTools (`scaffold`)
- [x] Package skeleton renamed to `EpiAwarePrototype` (fresh UUID
      `cbebd14a-101c-4997-a79f-d008ad7c07b2`)
- [x] Apache-2.0 LICENSE + NOTICE + attribution disclaimer in place
- [x] COMPLETE package ported onto `as_turing_model` (latest Turing) — nothing stubbed.
      Every upstream model, manipulator, modifier, ODE model, inference method, and
      the problem/method glue is ported. Validated on Turing 0.45 / DynamicPPL 0.41 /
      OrdinaryDiffEq 7 / Pathfinder 0.10, Julia 1.12. See "Ported surface" below.
- [x] Package loads; every ported model constructs + samples; composed models run NUTS.
      Composed `EpiAwareModel` and `EpiProblem` run rand/fix/condition/`|`/NUTS;
      Renewal and ODE (SIR) models fit under NUTS; the Pathfinder→NUTS `EpiMethod`
      runs end-to-end.
- [x] Full EpiAwareTestUtils test suite passes: 123 unit `@testitem`s + Aqua,
      ExplicitImports, docstring-format, doctest, formatting, JET (with a documented
      JET-runner workaround). Suite is green: 0 fail / 0 error / 0 broken. The
      package does NOT blanket-reexport Distributions/Turing (upstream did not
      either), so the docstring-format check only sees the package's own ~68
      documented names — no skipped/“broken” third-party names. Users
      `using EpiAwarePrototype, Distributions, Turing`.
- [~] Docs ported + decluttered — honest set seeded (getting started, composable
      design, API reference covering the full surface); full worked-example narrative
      still to expand.
- [x] Issues filed against EpiAwarePackageTools for any template gaps
      (`ISSUES_FOR_PACKAGETOOLS.md`: LICENSE override, JET runner, ExplicitImports
      `@reexport`, doctest `@meta` under TestItemRunner).
- [ ] Complete working port → branch protection added → switch to review-PR workflow
      (port works + green; gate to be applied by the human).

### Ported surface (loads, samples, tested)

- Base/utils (`base.jl`, `utils.jl`): `AbstractEpiAwareModel`, `as_turing_model`,
  `accumulate_scan` + `AbstractAccumulationStep`, `HalfNormal`, `SafePoisson`,
  `SafeNegativeBinomial`, `NegativeBinomialMeanClust`, double-censored
  `censored_pmf`/`censored_cdf`/`∫F`, `condition_model`, flat `show`.
- Latent (`latent.jl`, `latent_extra.jl`): `IID`, `HierarchicalNormal`,
  `RandomWalk`, `AR`, `MA`, `Intercept`, `FixedIntercept`, `Null`, `DiffLatentModel`,
  `TransformLatentModel`, `PrefixLatentModel`, `RecordExpectedLatent`,
  `CombineLatentModels`, `ConcatLatentModels`, `BroadcastLatentModel` +
  `RepeatEach`/`RepeatBlock` + `broadcast_dayofweek`/`broadcast_weekly`,
  `arma`/`arima`.
- Infections (`infections.jl`, `infections_extra.jl`): `EpiData`,
  `DirectInfections`, `ExpGrowthRate`, `Renewal` (+ renewal steps),
  `R_to_r`/`r_to_R`/`expected_Rt`.
- ODE (`ode.jl`): `SIRParams`, `SEIRParams`, `ODEProcess`.
- Observations (`observations.jl`, `observations_extra.jl`): `PoissonError`,
  `NegativeBinomialError`, `LatentDelay`, `Ascertainment` (+ `ascertainment_dayofweek`),
  `Aggregate`, `TransformObservationModel`, `PrefixObservationModel`,
  `RecordExpectedObs`, `StackObservationModels`.
- Composition (`compose.jl`): `EpiAwareModel`.
- Inference (`inference.jl`): `EpiProblem`, `EpiMethod`, `NUTSampler`,
  `ManyPathfinder`, `DirectSample`, `apply_method`, `manypathfinder`,
  `EpiAwareObservables`/`generated_observables`, `spread_draws`, `get_param_array`.

### Deliberate architecture replacements (not omissions)

- The per-concept generate functions (`generate_latent`, `generate_latent_infs`,
  `generate_observations`, `generate_epiaware`) and the deep `AbstractTuring*`
  hierarchy are replaced by the single `as_turing_model` + `AbstractEpiAwareModel`,
  per the brief. `prefix_submodel` is replaced by `DynamicPPL.prefix`.
- Pretty-printing: upstream's `EpiAwareBase/prettyprinting.jl` (a PrettyPrinting.jl
  tree display) is replaced by a dependency-free `Base.show(::MIME"text/plain",
  ::AbstractEpiAwareModel)` in `base.jl` that lists the component's fields as a
  tree — equivalent readable output without the heavy dep.
- `EpiObsModels/utils.jl`'s `generate_observation_kernel` (a sparse delay-kernel
  matrix builder) is NOT ported: it is unexported and called nowhere in upstream
  (dead code), and porting it would add a `SparseArrays` dependency solely for an
  unused function. No functional surface is affected. The mean-cluster negative
  binomial helper from the same file IS ported (`NegativeBinomialMeanClust`).

### Deviations forced by the upgraded ecosystem (documented in code)

- `ODEProcess` default solver is `AutoVern7(Rodas5P())` and drops the Bool
  `:verbose` option — OrdinaryDiffEq v7 removed `Rodas5` from the top level and
  no longer accepts a Bool `verbose`.
- `spread_draws` normalises the dotted `.iteration`/`.chain` columns current
  MCMCChains emits.
- The Pathfinder→NUTS warm-start runs NUTS from the default start: current Turing
  requires an `AbstractInitStrategy` for `initial_params` (the old vector form is
  gone) and Pathfinder no longer exposes `draws_transformed.value`; the Pathfinder
  pre-step still runs and its result is returned.

### Censoring via CensoredDistributions.jl (done)

- Per the brief, the bespoke double-interval-censoring code is dropped and censoring
  is delegated to **CensoredDistributions.jl**. The internal `_discretised_pmf`
  helper (in `utils.jl`) builds the right-truncated discrete PMF with
  `double_interval_censored(dist; upper = D, interval = Δd)`, evaluating its `pdf`
  on the bin left-edges and normalising. The two call sites — `EpiData(;
  gen_distribution = …)` and `LatentDelay(model, distribution)` — use it. The
  exported `censored_pmf`/`censored_cdf`/`∫F` are removed (users wanting censoring
  use CensoredDistributions directly).
- **UUID caution resolved.** CensoredDistributions.jl is registered in General under
  UUID `b2eeebe4-...` (the same UUID upstream EpiAware once used). Because
  EpiAwarePrototype has its own distinct UUID (`cbebd14a-…`) and does NOT depend on
  the upstream EpiAware package, the env resolves CensoredDistributions cleanly from
  the registry with no clash. Verified: added, freed to the registered version
  (0.2.21), suite green.
