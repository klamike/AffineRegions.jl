# Examples Environment

This directory has its own Julia environment so larger example-only dependencies
do not leak into the main `AffineRegions.jl` package environment.

From this directory, instantiate the environment:

```julia
using Pkg
Pkg.instantiate()
```

The committed `Project.toml` already points at the local parent
[`AffineRegions.jl`](/Users/mklamkin3/Git/BatchLP/AffineRegions.jl) checkout and the
local sibling [`DiffOpt.jl`](/Users/mklamkin3/Git/BatchLP/DiffOpt.jl) checkout through
`[sources]`.

Then run the larger PEGASE example with:

```bash
julia --project=. dcopf_89_pegase.jl
```

To benchmark atlas serving against `POI + HiGHS` on the same PEGASE setup, run:

```bash
julia --project=. benchmark_poi_highs_vs_atlas.jl
```

The main package test suite continues to use the smaller self-contained example
in the parent package environment.
