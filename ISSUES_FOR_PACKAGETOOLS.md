# Issues to file against EpiAware/EpiAwarePackageTools.jl

These are template/scaffold gaps encountered while adopting EpiAwarePackageTools
in EpiAwarePrototype.jl. Each is a candidate GitHub issue. The minimal local
workaround actually applied is noted under each.

**Filed:**
- #1 (MIT LICENSE / managed-file revert) → already tracked upstream as
  [EpiAwarePackageTools#14](https://github.com/EpiAware/EpiAwarePackageTools.jl/issues/14)
- #2 (JET vs DynamicPPL `@model`) → [#16](https://github.com/EpiAware/EpiAwarePackageTools.jl/issues/16)
- #3 (`test_explicit_imports` vs `@reexport`) → [#17](https://github.com/EpiAware/EpiAwarePackageTools.jl/issues/17)
- #4 (`test_doctest` `@meta` under TestItemRunner) → [#18](https://github.com/EpiAware/EpiAwarePackageTools.jl/issues/18)

---

## 1. `scaffold` writes an MIT `LICENSE`; no way to request a different licence

**Expected:** A package that must ship under Apache-2.0 (e.g. because it
incorporates Apache-2.0 code) can scaffold without having its `LICENSE`
overwritten with MIT, or can tell `scaffold`/`update` which licence to write.

**What happened:** `LICENSE` is a MANAGED template
(`Template("LICENSE", "LICENSE", true, true)` in `src/scaffold.jl`) hardcoded to
the MIT text in `templates/LICENSE`. `scaffold` always (re)writes it, and there
is no `license` keyword in `scaffold_inputs`. Worse, because `LICENSE` is
*managed*, the scheduled `update(pkgdir)` template-sync will silently revert any
package that replaces it with a different licence — re-introducing a licence
incompatibility on every sync.

**Minimal repro:**
```julia
using EpiAwarePackageTools
scaffold("/path/to/pkg")          # writes templates/LICENSE (MIT)
# replace LICENSE with Apache-2.0 by hand
update("/path/to/pkg")            # reverts it back to MIT
```

**Suggested fix:** add a `license` input to `scaffold_inputs` (default `"MIT"`)
selecting among bundled `templates/LICENSE.<spdx>` files, and/or treat `LICENSE`
as package-owned (write-once) rather than managed so a deliberate licence choice
is never reverted by a sync.

**Local workaround applied:** overwrote `LICENSE` with the upstream Apache-2.0
text after scaffolding, added a `NOTICE`, and recorded the override here. The
managed-file revert risk above remains until this is fixed upstream — anyone
running `update()` on this package must re-apply the Apache-2.0 `LICENSE`.

---

## 2. JET runner cannot analyse a DynamicPPL `@model` package cleanly

**Expected:** A Turing/`DynamicPPL` package (whose core surface is `@model`
functions) can pass the scaffolded JET check, since `@model` is the normal way
to write such a package.

**What happened:** The managed `test/jet/runtests.jl` calls
`JET.report_package(mod; target_modules = (mod,))` and fails on any report. For
a `@model` package this always fails: JET emits a
`local variable `x` is not defined` (`UndefVarErrorReport`) for *every*
`~`-assigned variable, because the tilde macro hides the assignment from JET's
static analysis. Our package gets 19 such false positives (`ar_init`, `ϵ_t`,
`damp_AR`, `priors`, `Z_t`, `I_t`, ... — all `~` targets), none of which is a
real defect (the models sample correctly under NUTS). Notably, the package this
prototype is adapted from avoided JET entirely and used only Aqua, which
suggests this incompatibility is long-standing.

**Minimal repro:** scaffold any package whose public functions are Turing
`@model`s and run `julia --project=test/jet test/jet/runtests.jl`.

**Suggested fix:** give the JET helper / managed runner a hook to filter
reports — e.g. a `report_filter` callback, or built-in suppression of
`UndefVarErrorReport`s whose enclosing `MethodInstance` takes the DynamicPPL
evaluator signature `(::DynamicPPL.Model, ::DynamicPPL.AbstractVarInfo, ...)`.
Alternatively allow `qa_config.jl` to disable the JET testset for `@model`
packages.

**Local workaround applied:** replaced the managed `test/jet/runtests.jl` with a
version that runs `report_package` and drops every report arising inside a
`@model`-generated method (matched on the DynamicPPL evaluator signature
`(::Model, ::AbstractVarInfo, ...)`), failing on any *other* report. This covers
both classes of macro artifact: `UndefVarErrorReport`s for `~`-assigned locals
and `MethodErrorReport`s through the `:=` (coloneq) tracking machinery
(`store_coloneq_value!!`). Added `DynamicPPL` to `test/jet/Project.toml` (within
the template's stated "add packages JET needs" allowance). A template-sync will
revert the runner; it must be re-applied until the helper supports a filter.

---

## 3. `test_explicit_imports` un-ignorable `check_no_implicit_imports` breaks `@reexport`

**Expected:** A package that reexports a dependency with
`@reexport using SomePkg` (a standard, recommended pattern, used by the package
this prototype is adapted from) can pass `test_explicit_imports`.

**What happened:** `test_explicit_imports(mod; ignore)` forwards `ignore` only to
`check_all_explicit_imports_are_public`, **not** to `check_no_implicit_imports`.
`@reexport using Turing` makes the bare module name `Turing` an implicit import,
which `check_no_implicit_imports` flags with no way to ignore it via the helper.

**Minimal repro:** a package with `@reexport using Turing` (and which does not
otherwise explicitly `using Turing: Turing`) calling `test_explicit_imports`.

**Suggested fix:** forward an `implicit_ignore` (or reuse `ignore`) to
`check_no_implicit_imports` in `test_explicit_imports`.

**Local workaround applied (then superseded):** initially added
`using Turing: Turing, ...` so the reexported module names were also explicit
imports. This was later made moot: the prototype stopped blanket-reexporting
Distributions/Turing altogether (the upstream EpiAware did not reexport them
either; users `using EpiAwarePrototype, Distributions, Turing`). That also drove
the docstring-format check from ~326 skipped/"broken" third-party names to zero.
The underlying helper limitation still stands for any package that *does* want to
`@reexport`, so the issue remains worth fixing upstream.

---

## 4. `test_doctest` + `@meta CurrentModule` fails under TestItemRunner isolation

**Expected:** `@meta CurrentModule = MyPackage` blocks in `docs/src/*.md` (the
standard Documenter idiom) work with the scaffolded `test_doctest` testset.

**What happened:** `test_doctest(mod)` calls `Documenter.doctest(mod)`, which
evaluates each page's `@meta CurrentModule = MyPackage` block against `Main`.
Run standalone this is fine (the REPL/`make.jl` has `using MyPackage`), but under
`TestItemRunner` the `@testitem` body runs in a sandbox module, so `Main` has no
`MyPackage` binding and every `@meta` block errors with
`UndefVarError: MyPackage not defined in Main`, failing the doctest testset.

**Minimal repro:** add `@meta CurrentModule = MyPackage` to a `docs/src` page and
run the scaffolded `test/runtests.jl` (which drives the doctest testitem via
TestItemRunner).

**Suggested fix:** have `test_doctest` import the target module into the doctest
evaluation `Main` (or pass a `Module` to `Documenter.doctest`) so `@meta` blocks
resolve under TestItemRunner.

**Local workaround applied:** removed the `@meta CurrentModule` blocks from the
docs pages. `Documenter.doctest` skips template expansion so it does not need
them, and `docs/make.jl` uses exported names plus `setdocmeta!` for the full
build's cross-references.
