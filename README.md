# AffineRegions.jl

`AffineRegions.jl` builds and queries basis certificates for parametric LPs of the form

\[
\min_x (c + H p)^\top x + c_0
\]
\[
\text{s.t. } A x + F p \le b
\]

with support for equality constraints, free variables, lower bounds, and upper bounds through an internal standard-form reduction.

The package is optimized for repeated solves on realistic, degenerate LPs:

- certificate construction uses native basis solves and basis-transpose solves
- region queries are dense matrix products over packed atlases
- primal-law clustering prevents duplicate degenerate regions from bloating the atlas
- cached optimizer hits reconstruct primal and dual solutions without re-solving
- miss-path solves can warm-start HiGHS from the last region basis

## Status

Current native basis backend support:

- `HiGHS`
- optional `Gurobi` extension via `ext/AffineRegionsGurobiExt.jl`

The solver-extension boundary is explicit in the code, so additional native-basis backends such as `Gurobi` can be added without changing the public atlas/query API.

## Primary API

The intended production flow is atlas-first:

```julia
using AffineRegions
using JuMP, DiffOpt, HiGHS

model = DiffOpt.basis_diff_model(HiGHS.Optimizer)
set_silent(model)

@variable(model, p in Parameter(1.0))
@variable(model, x >= 0)
@constraint(model, x >= p)
@constraint(model, x >= -p)
@objective(model, Min, x)

atlas = RegionAtlas()

for pval in (-1.0, 1.0)
    set_parameter_value(p, pval)
    optimize!(model)
    add_certificate!(atlas, certificate(model))
end

frozen = freeze(atlas)
pts = reshape([-0.5, 0.0, 0.5], 1, :)

ids = query_batch(frozen, pts)
result = solve_batch(frozen, pts)
```

Use this path when region identity is not known in advance, which is the normal case for deployment.

## Atlas API

`RegionAtlas` is the main public data structure and `FrozenAtlas` is the preferred query-only form.

Use `FrozenAtlas` for concurrent query workloads. It has no mutable query cache and is the concurrency-safe target for shared inference.

- `add_certificate!`
- `query`
- `query_batch`
- `solve_batch`
- `contains_batch`
- `primal_batch`
- `dual_batch`
- `freeze`
- `thaw`
- `atlas_stats`
- `atlas_metrics`
- `reset_metrics!`
- `compact!`
- `remove_certificate!`
- `clear!`

The highest-throughput CPU path is the packed atlas query engine underneath `FrozenAtlas`.

## Offline Atlas Building

The package now has an explicit offline atlas builder for precomputing a serving artifact:

```julia
samples = reshape([-2.0, -1.0, 0.0, 1.0, 2.0], 1, :)
build = build_atlas(model, samples; freeze_result=true)

frozen = build.atlas
report = build.report
```

`build_atlas!` appends into an existing mutable atlas, while `build_atlas` creates a fresh atlas and optionally returns a frozen query-only result.

A small self-contained DCOPF example is available in [examples/dcopf_parametric_load.jl](/Users/mklamkin3/Git/BatchLP/AffineRegions.jl/examples/dcopf_parametric_load.jl).

Larger power-system examples live in the separate environment under [examples/README.md](/Users/mklamkin3/Git/BatchLP/AffineRegions.jl/examples/README.md). In particular, [examples/dcopf_89_pegase.jl](/Users/mklamkin3/Git/BatchLP/AffineRegions.jl/examples/dcopf_89_pegase.jl) builds a parametric DCOPF on the real `89_pegase` PGLib case using the same formulation structure as `PGLearn`'s DCOPF model.

## Certificate API

Single-certificate operations are still available, but they are lower-level primitives. They are most useful for:

- certificate construction and debugging
- validation and tests
- workloads that have already been partitioned by region

Build a certificate from a solved parametric model:

```julia
using AffineRegions
using JuMP, DiffOpt, HiGHS

model = DiffOpt.basis_diff_model(HiGHS.Optimizer)
set_silent(model)

@variable(model, p in Parameter(1.0))
@variable(model, x >= 0)
@constraint(model, x >= p)
@objective(model, Min, x)

optimize!(model)

cert = certificate(model)
```

The certificate stores:

- `C, c` with region law `C * p + c >= 0`
- `P, p0` with primal law `x = P * p + p0`
- `D, d0` with a basis-specific dual law for the original model

Evaluate it directly:

```julia
p = [1.0]
contains_point(cert, p)
primal_solution(cert, p)
dual_solution(cert, p)
```

## Persistence

Atlases can be snapshotted and restored without rebuilding regions:

```julia
save_atlas("atlas.bin", atlas)
loaded = load_atlas("atlas.bin")
loaded_frozen = load_atlas("atlas.bin"; frozen=true)
```

Persistence now uses a versioned stable tar archive with TOML metadata plus raw numeric blobs. `load_atlas` still accepts the older Julia-`Serialization` snapshot files as a legacy fallback.

Atlas persistence includes both the numerical policy and atlas policy so a frozen serving artifact can be reloaded with the same eviction/learning configuration.

## Public Batch API

The production query path is the atlas batch API:

- `contains_batch`
- `query_batch`
- `solve_batch`
- `primal_batch`
- `dual_batch`
- `contains_batch!`
- `query_batch!`
- `primal_batch!`
- `dual_batch!`

These functions operate on `RegionAtlas`, `FrozenAtlas`, or `PackedRegionAtlas`. For most users, `FrozenAtlas` should be the default target.

For GPU-oriented deployments, `gpu_pack(atlas; T=Float32, I=Int32)` exports the same atlas into a typed, device-friendly representation `GPUPackedRegionAtlas`.

## Optimizer Wrapper

The main end-user wrapper is:

```julia
fast = AffineRegions.Model(HiGHS.Optimizer)
```

It behaves like a JuMP optimizer, but:

- checks cached regions first
- reconstructs cached primal and dual solutions directly on hits
- can opt into native basis hints on misses with `use_basis_hints=true`

The wrapper is convenient and fast, but the highest-throughput path is still the explicit atlas batch API.

## Numerical Policy

Tolerance and reduction behavior is centralized in `NumericalPolicy`:

```julia
policy = NumericalPolicy(
    1e-10,  # zero_tol
    1e-8,   # basis_tol
    1e-8,   # region_tol
    1e-9,   # reduction_tol
    1e-8,   # dedup_tol
    false,  # exact_reduction
    8,      # exact_reduction_param_limit
    256,    # exact_reduction_row_limit
)

atlas = RegionAtlas(; policy)
cert = certificate(model; policy)
```

Atlas growth and learning behavior is controlled separately through `AtlasPolicy`:

```julia
atlas_policy = AtlasPolicy(
    max_certificates=10_000,
    max_shared_rows=250_000,
    learning_enabled=true,
    eviction=:lru,
)

atlas = RegionAtlas(; policy, atlas_policy)
```

The package also tracks production-facing counters such as query hits, misses, batch calls, builder progress, dedup hits, and evictions through `atlas_metrics(atlas_like)`.

## Failure Modes

The package throws explicit exceptions for unsupported or invalid states:

- `UnsupportedModelError`
- `InvalidModelStateError`
- `NativeBasisError`
- `QueryCoverageError`

Important contract points:

- the model must be solved and feasible before `certificate(model)`
- linear minimization and maximization LPs are supported
- the returned dual law is basis-specific under degeneracy
- `solve_batch` throws `QueryCoverageError` if any point is uncovered and `require_coverage=true`

## Performance Model

The implementation is designed around GPU-friendly operations:

- packed region checks are `C * Pbatch + c`
- primal recovery is `P * Pbatch + p0`
- dual recovery is `D * Pbatch + d0`

The current implementation is CPU-only, but the atlas layout is intentionally close to the shape needed for future device kernels.

## Validation

The repository includes:

- correctness tests in `test/runtests.jl`
- realistic CPU benchmarks in `test/benchmark.jl`
- CI perf smoke in `test/perf_smoke.jl`

Run locally:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. test/perf_smoke.jl
julia --project=. test/benchmark.jl
```
