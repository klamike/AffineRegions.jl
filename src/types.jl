struct AffineRegion
    constraints::Vector{JuMP.ScalarConstraint}
    primal_law::Dict{JuMP.VariableRef,JuMP.AffExpr}
    dual_law::Dict{JuMP.ConstraintRef,JuMP.AffExpr}
end

struct VarRep
    offset::Float64
    cols::Vector{Int}
    vals::Vector{Float64}
end

struct LPData
    variables::Vector{JuMP.VariableRef}
    parameters::Vector{JuMP.VariableRef}
    duals::Vector{JuMP.ConstraintRef}
    obj_c::Vector{Float64}
    obj_H::Matrix{Float64}
    obj_c0::Float64
    obj_scale::Float64
    row_matrix::SparseArrays.SparseMatrixCSC{Float64,Int}
    row_signs::Vector{Float64}
    dual_kinds::Vector{UInt8}
    dual_sources::Vector{Int}
    A::SparseArrays.SparseMatrixCSC{Float64,Int}
    F::Matrix{Float64}
    rhs_parametric::Bool
    b::Vector{Float64}
    c::Vector{Float64}
    H::Matrix{Float64}
    obj_parametric::Bool
    c0::Float64
    x_offset::Vector{Float64}
    x_map::SparseArrays.SparseMatrixCSC{Float64,Int}
    var_reps::Vector{VarRep}
    optimizer_constructor::Any
end

struct BasisCertificate
    basis_cols::Vector{Int}
    anchor_p::Vector{Float64}
    C::Matrix{Float64}
    c::Vector{Float64}
    P::Matrix{Float64}
    p0::Vector{Float64}
    D::Matrix{Float64}
    d0::Vector{Float64}
    var_basis::Vector{MOI.BasisStatusCode}
    row_basis::Vector{MOI.BasisStatusCode}
    basis_hint::NativeBasisHint
end

const MatrixAffineRegion = BasisCertificate

mutable struct CanonicalSolveCache
    model::JuMP.Model
    z::Vector{JuMP.VariableRef}
    eq_refs::Vector{JuMP.ConstraintRef}
    rhs::Vector{Float64}
    basis_rhs::Vector{Float64}
    basis_sol::Vector{Float64}
    col_status::Vector{HiGHS.HighsInt}
    row_status::Vector{HiGHS.HighsInt}
    basic_vars::Vector{Cint}
    z0::Vector{Float64}
    Zp::Matrix{Float64}
    y0::Vector{Float64}
    Yp::Matrix{Float64}
end

struct PackedRegionAtlas
    C::Matrix{Float64}
    c::Vector{Float64}
    row_offsets::Vector{Int}
    P::Array{Float64,3}
    p0::Matrix{Float64}
    D::Array{Float64,3}
    d0::Matrix{Float64}
end

struct GPUPackedRegionAtlas{T<:AbstractFloat,I<:Integer}
    C::Matrix{T}
    c::Vector{T}
    row_offsets::Vector{I}
    P::Array{T,3}
    p0::Matrix{T}
    D::Array{T,3}
    d0::Matrix{T}
end

mutable struct ConstraintStore
    C::Vector{Vector{Float64}}
    c::Vector{Float64}
    refcount::Vector{Int}
    index::Dict{String,Int}
end

mutable struct AtlasMetrics
    query_calls::Base.Threads.Atomic{Int}
    query_hits::Base.Threads.Atomic{Int}
    query_misses::Base.Threads.Atomic{Int}
    batch_query_calls::Base.Threads.Atomic{Int}
    certificates_added::Base.Threads.Atomic{Int}
    certificate_dedup_hits::Base.Threads.Atomic{Int}
    region_dedup_hits::Base.Threads.Atomic{Int}
    cluster_reuses::Base.Threads.Atomic{Int}
    evictions::Base.Threads.Atomic{Int}
    policy_rejections::Base.Threads.Atomic{Int}
    pack_rebuilds::Base.Threads.Atomic{Int}
    builder_points::Base.Threads.Atomic{Int}
    builder_feasible::Base.Threads.Atomic{Int}
    builder_skipped::Base.Threads.Atomic{Int}
end

struct AtlasMetricsSnapshot
    query_calls::Int
    query_hits::Int
    query_misses::Int
    batch_query_calls::Int
    certificates_added::Int
    certificate_dedup_hits::Int
    region_dedup_hits::Int
    cluster_reuses::Int
    evictions::Int
    policy_rejections::Int
    pack_rebuilds::Int
    builder_points::Int
    builder_feasible::Int
    builder_skipped::Int
end

mutable struct RegionAtlas
    certificates::Vector{BasisCertificate}
    certificate_rows::Vector{Vector{Int}}
    signatures::Dict{String,Int}
    region_signatures::Dict{String,Int}
    primal_signatures::Dict{String,Int}
    cluster_members::Vector{Vector{Int}}
    certificate_clusters::Vector{Int}
    certificate_representatives::Vector{Int}
    constraint_store::ConstraintStore
    last_hit::Int
    packed::Union{Nothing,PackedRegionAtlas}
    packed_dirty::Bool
    policy::NumericalPolicy
    atlas_policy::AtlasPolicy
    metrics::AtlasMetrics
    last_access::Vector{Int}
    insertion_order::Vector{Int}
    access_clock::Int
    frozen::Bool
end

mutable struct PackedAtlasWorkspace
    vals::Matrix{Float64}
    hits::BitMatrix
    cert_ids::Vector{Int}
    counts::Vector{Int}
    offsets::Vector{Int}
    perm::Vector{Int}
end

struct AtlasStats
    n_certificates::Int
    n_primal_clusters::Int
    n_region_representatives::Int
    total_region_rows::Int
    shared_region_rows::Int
    compression_ratio::Float64
    parameter_dimension::Int
    primal_dimension::Int
    dual_dimension::Int
    packed_rows::Int
end

struct FrozenAtlas
    certificates::Vector{BasisCertificate}
    packed::PackedRegionAtlas
    policy::NumericalPolicy
    atlas_policy::AtlasPolicy
    stats::AtlasStats
    metrics::AtlasMetrics
end

struct AtlasSnapshot
    format_version::Int
    created_at::String
    certificates::Vector{BasisCertificate}
    certificate_rows::Vector{Vector{Int}}
    constraint_C::Vector{Vector{Float64}}
    constraint_c::Vector{Float64}
    constraint_refcount::Vector{Int}
    last_hit::Int
    policy::NumericalPolicy
    atlas_policy::AtlasPolicy
end

struct BatchResult
    ids::Vector{Int}
    primal::Matrix{Float64}
    dual::Matrix{Float64}
end

struct AtlasBuildReport
    attempted_points::Int
    feasible_points::Int
    skipped_points::Int
    added_certificates::Int
    deduplicated_points::Int
    final_certificates::Int
    frozen::Bool
end

ConstraintStore() = ConstraintStore(Vector{Vector{Float64}}(), Float64[], Int[], Dict{String,Int}())

AtlasMetrics() = AtlasMetrics(
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
    Base.Threads.Atomic{Int}(0),
)

RegionAtlas(; policy=default_policy(), atlas_policy=default_atlas_policy(), frozen=false) =
    RegionAtlas(
        BasisCertificate[],
        Vector{Vector{Int}}(),
        Dict{String,Int}(),
        Dict{String,Int}(),
        Dict{String,Int}(),
        Vector{Vector{Int}}(),
        Int[],
        Int[],
        ConstraintStore(),
        0,
        nothing,
        true,
        policy,
        atlas_policy,
        AtlasMetrics(),
        Int[],
        Int[],
        0,
        frozen,
    )

function RegionAtlas(certificates::Vector{BasisCertificate}, last_hit::Int=0; policy=default_policy(), atlas_policy=default_atlas_policy(), frozen=false)
    atlas = RegionAtlas(; policy, atlas_policy, frozen)
    for cert in certificates
        add_certificate!(atlas, cert)
    end
    atlas.last_hit = isempty(atlas.certificates) ? 0 : min(last_hit, length(atlas.certificates))
    return atlas
end
