module AffineRegions

import Dates
import DiffOpt
import HiGHS
import JuMP
import LinearAlgebra
import ParametricOptInterface as POI
import Serialization
import SparseArrays
import TOML
import Tar

const MOI = JuMP.MOI

export affine_region, matrix_affine_region, certificate, extract_lp
export contains_point, contains_batch, primal_solution, primal_batch, dual_solution, dual_batch, query
export pack, query_batch, PackedRegionAtlas, PackedAtlasWorkspace, add_certificate!
export query_batch!, contains_batch!, primal_batch!, dual_batch!, solve_batch, BatchResult
export gpu_pack, GPUPackedRegionAtlas
export freeze, thaw, FrozenAtlas, AtlasSnapshot, save_atlas, load_atlas, snapshot
export atlas_stats, atlas_metrics, reset_metrics!, compact!, remove_certificate!, clear!
export build_atlas, build_atlas!, AtlasBuildReport
export NumericalPolicy, default_policy, AtlasPolicy, default_atlas_policy, AtlasStats, AtlasMetricsSnapshot
export AffineRegionsError, UnsupportedModelError, InvalidModelStateError, NativeBasisError, QueryCoverageError
export AffineRegion, BasisCertificate, MatrixAffineRegion, LPData, RegionAtlas, Model

include("errors.jl")
include("helpers.jl")
include("types.jl")
include("extraction.jl")
include("basis.jl")
include("api.jl")
include("atlas.jl")
include("builder.jl")
include("persistence.jl")
include("gpu_pack.jl")
include("optimizer.jl")

end # module AffineRegions
