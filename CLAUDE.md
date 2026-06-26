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
- **Do NOT reference any private repository in any package-facing file, commit
  message, issue, or docs page.** In particular, never name or link the private
  "case for composability" / composability paper repository. Attribute **only**
  the open Apache-2.0 code above. Doc content drawn from internal sources must be
  rewritten in our own words as original package documentation.

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

  The second arg `false` disables automatic variable prefixing (preserve
  existing flat naming unless prefixing is explicitly required). Replace the
  old `prefix_submodel` helper with `to_submodel`-based prefixing.

## Naming

- Package name and top-level module: **`EpiAwarePrototype`** (fresh UUID in
  `Project.toml`). Update all references, docstrings, doctests, and CI badges.

## Docs

- Make it clear throughout that this is a **prototype for composable infectious
  disease modelling**.
- Port the composable-modelling narrative (the DSL idea + the Turing.jl backend /
  `as_turing_model` design) into the package docs, **rewritten as original
  package documentation** — adapt to the new API.
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
"never reference the private repo", commit-to-`main`, and issue-logging rules are
*how* the port is done — they are not a reduction of scope.

Sequence the work so `main` stays loadable (port in dependency order, commit in
logical chunks). The deliverable is the **complete, working package**: it loads, every
ported model can be constructed and sampled, end-to-end composed models run
(rand/fix/condition/NUTS), and the full EpiAwareTestUtils test suite passes. Only
switch to the review-PR workflow once that complete version works.

## Current status

- [ ] Repo scaffolded from EpiAwarePackageTools (`scaffold`)
- [ ] Package skeleton renamed to `EpiAwarePrototype` (fresh UUID)
- [ ] Apache-2.0 LICENSE + NOTICE + attribution disclaimer in place
- [ ] COMPLETE package ported onto `as_turing_model` (latest Turing) — nothing stubbed
- [ ] Package loads; every ported model constructs + samples; composed models run NUTS
- [ ] Full EpiAwareTestUtils test suite passes
- [ ] Docs ported + decluttered
- [ ] Issues filed against EpiAwarePackageTools for any template gaps
- [ ] Complete working port → branch protection added → switch to review-PR workflow
