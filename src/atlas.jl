_metric_fields() = (
    :query_calls,
    :query_hits,
    :query_misses,
    :batch_query_calls,
    :certificates_added,
    :certificate_dedup_hits,
    :region_dedup_hits,
    :cluster_reuses,
    :evictions,
    :policy_rejections,
    :pack_rebuilds,
    :builder_points,
    :builder_feasible,
    :builder_skipped,
)

_metric_inc!(metrics::AtlasMetrics, field::Symbol, delta::Integer=1) =
    Base.Threads.atomic_add!(getfield(metrics, field), Int(delta))

_metric_get(metrics::AtlasMetrics, field::Symbol) = getfield(metrics, field)[]

function reset_metrics!(atlas_like)
    metrics = atlas_like.metrics
    for field in _metric_fields()
        Base.Threads.atomic_xchg!(getfield(metrics, field), 0)
    end
    return atlas_like
end

function atlas_metrics(atlas_like)
    metrics = atlas_like.metrics
    return AtlasMetricsSnapshot((_metric_get(metrics, field) for field in _metric_fields())...)
end

function _touch_certificate!(atlas::RegionAtlas, idx::Integer)
    1 <= idx <= length(atlas.certificates) || return atlas
    atlas.access_clock += 1
    atlas.last_access[idx] = atlas.access_clock
    atlas.last_hit = idx
    return atlas
end

function _touch_certificate!(frozen::FrozenAtlas, _idx::Integer)
    return frozen
end

function _record_query_metrics!(atlas_like, ids::AbstractVector{Int})
    _metric_inc!(atlas_like.metrics, :batch_query_calls)
    _metric_inc!(atlas_like.metrics, :query_calls, length(ids))
    hits = count(!iszero, ids)
    _metric_inc!(atlas_like.metrics, :query_hits, hits)
    _metric_inc!(atlas_like.metrics, :query_misses, length(ids) - hits)
    return atlas_like
end

function _select_eviction_candidate(atlas::RegionAtlas)
    isempty(atlas.certificates) && return 0
    atlas.atlas_policy.eviction == :none && return 0
    if atlas.atlas_policy.eviction == :fifo
        best_idx, best_order = 1, atlas.insertion_order[1]
        for idx in 2:length(atlas.insertion_order)
            order = atlas.insertion_order[idx]
            if order < best_order
                best_idx, best_order = idx, order
            end
        end
        return best_idx
    end
    best_idx, best_access = 1, atlas.last_access[1]
    for idx in 2:length(atlas.last_access)
        access = atlas.last_access[idx]
        if access < best_access
            best_idx, best_access = idx, access
        end
    end
    return best_idx
end

function _enforce_atlas_policy!(atlas::RegionAtlas)
    while length(atlas.certificates) > atlas.atlas_policy.max_certificates ||
          length(atlas.constraint_store.c) > atlas.atlas_policy.max_shared_rows
        victim = _select_eviction_candidate(atlas)
        if victim == 0
            _metric_inc!(atlas.metrics, :policy_rejections)
            remove_certificate!(atlas, length(atlas.certificates))
            return atlas
        end
        _metric_inc!(atlas.metrics, :evictions)
        remove_certificate!(atlas, victim)
    end
    return atlas
end

function query(atlas::RegionAtlas, p::AbstractVector; policy=atlas.policy, tol=policy.region_tol)
    _metric_inc!(atlas.metrics, :query_calls)
    if atlas.last_hit > 0 && atlas.last_hit <= length(atlas.certificates)
        cert = atlas.certificates[atlas.last_hit]
        if contains_point(cert, p; tol)
            _metric_inc!(atlas.metrics, :query_hits)
            _touch_certificate!(atlas, atlas.last_hit)
            return cert
        end
    end
    for (i, cert) in enumerate(atlas.certificates)
        i == atlas.last_hit && continue
        if contains_point(cert, p; tol)
            _metric_inc!(atlas.metrics, :query_hits)
            _touch_certificate!(atlas, i)
            return cert
        end
    end
    _metric_inc!(atlas.metrics, :query_misses)
    return nothing
end

function add_certificate!(atlas::RegionAtlas, cert::BasisCertificate)
    atlas.frozen && throw(InvalidModelStateError("Cannot mutate a frozen RegionAtlas. Create a mutable atlas with thaw(frozen_atlas) first."))
    if !atlas.atlas_policy.learning_enabled
        _metric_inc!(atlas.metrics, :policy_rejections)
        return atlas
    end
    sig = _certificate_signature(cert; tol=atlas.policy.dedup_tol)
    if haskey(atlas.signatures, sig)
        _metric_inc!(atlas.metrics, :certificate_dedup_hits)
        _touch_certificate!(atlas, atlas.signatures[sig])
        return atlas
    end
    region_sig = _region_signature(cert; tol=atlas.policy.dedup_tol)
    if haskey(atlas.region_signatures, region_sig)
        _metric_inc!(atlas.metrics, :region_dedup_hits)
        _touch_certificate!(atlas, atlas.region_signatures[region_sig])
        return atlas
    end
    row_ids = _store_constraints!(atlas.constraint_store, cert.C, cert.c; tol=atlas.policy.dedup_tol)
    cluster_sig = _primal_cluster_signature(cert; tol=atlas.policy.dedup_tol)
    reused_cluster = haskey(atlas.primal_signatures, cluster_sig)
    cluster_id = get!(atlas.primal_signatures, cluster_sig) do
        push!(atlas.cluster_members, Int[])
        return length(atlas.cluster_members)
    end
    push!(atlas.certificates, cert)
    push!(atlas.certificate_rows, row_ids)
    cert_idx = length(atlas.certificates)
    push!(atlas.certificate_clusters, cluster_id)
    push!(atlas.certificate_representatives, cert_idx)
    push!(atlas.cluster_members[cluster_id], cert_idx)
    atlas.signatures[sig] = cert_idx
    atlas.region_signatures[region_sig] = cert_idx
    atlas.access_clock += 1
    push!(atlas.last_access, atlas.access_clock)
    push!(atlas.insertion_order, atlas.access_clock)
    atlas.last_hit = cert_idx
    atlas.packed = nothing
    atlas.packed_dirty = true
    reused_cluster && _metric_inc!(atlas.metrics, :cluster_reuses)
    _metric_inc!(atlas.metrics, :certificates_added)
    return _enforce_atlas_policy!(atlas)
end

function pack(atlas::RegionAtlas; force=false)
    if !force && !atlas.packed_dirty && atlas.packed !== nothing
        return atlas.packed
    end
    isempty(atlas.certificates) && error("Cannot pack an empty RegionAtlas.")
    cert0 = atlas.certificates[1]
    k = size(cert0.C, 2)
    n_x = size(cert0.P, 1)
    n_d = size(cert0.D, 1)
    n_cert = length(atlas.certificates)
    total_rows = sum(length(rows) for rows in atlas.certificate_rows)

    C = zeros(Float64, total_rows, k)
    c = zeros(Float64, total_rows)
    row_offsets = Vector{Int}(undef, n_cert + 1)
    row_offsets[1] = 1
    P = zeros(Float64, n_x, k, n_cert)
    p0 = zeros(Float64, n_x, n_cert)
    D = zeros(Float64, n_d, k, n_cert)
    d0 = zeros(Float64, n_d, n_cert)

    next_row = 1
    for (idx, cert) in enumerate(atlas.certificates)
        size(cert.C, 2) == k || error("All certificates in a packed atlas must share parameter dimension.")
        size(cert.P, 1) == n_x || error("All certificates in a packed atlas must share primal dimension.")
        size(cert.D, 1) == n_d || error("All certificates in a packed atlas must share dual dimension.")

        rows = length(atlas.certificate_rows[idx])
        if rows > 0
            row_range = next_row:(next_row + rows - 1)
            _load_constraint_rows!(C, c, row_range, atlas.constraint_store, atlas.certificate_rows[idx])
            next_row += rows
        end
        row_offsets[idx + 1] = next_row
        P[:, :, idx] .= cert.P
        p0[:, idx] .= cert.p0
        D[:, :, idx] .= cert.D
        d0[:, idx] .= cert.d0
    end

    atlas.packed = PackedRegionAtlas(C, c, row_offsets, P, p0, D, d0)
    atlas.packed_dirty = false
    _metric_inc!(atlas.metrics, :pack_rebuilds)
    return atlas.packed
end

pack(frozen::FrozenAtlas) = frozen.packed

function PackedAtlasWorkspace(atlas::PackedRegionAtlas, batch_size::Integer)
    n_rows = size(atlas.C, 1)
    n_cert = size(atlas.P, 3)
    return PackedAtlasWorkspace(
        zeros(Float64, n_rows, batch_size),
        falses(n_cert, batch_size),
        zeros(Int, batch_size),
        zeros(Int, n_cert),
        zeros(Int, n_cert + 1),
        zeros(Int, batch_size),
    )
end

function contains_batch(atlas::PackedRegionAtlas, Pbatch::AbstractMatrix; tol=1e-8, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    n_cert = length(atlas.row_offsets) - 1
    n_pts = size(Pbatch, 2)
    hits = workspace === nothing ? falses(n_cert, n_pts) : _workspace_hits(workspace, atlas, n_pts)
    vals = _packed_constraint_values(atlas, Pbatch; workspace)
    for idx in 1:n_cert
        row_start = atlas.row_offsets[idx]
        row_stop = atlas.row_offsets[idx + 1] - 1
        if row_stop < row_start
            hits[idx, :] .= true
            continue
        end
        hits[idx, :] .= vec(all(@view(vals[row_start:row_stop, :]) .>= -tol, dims=1))
    end
    return hits
end

function contains_batch(atlas::RegionAtlas, Pbatch::AbstractMatrix; tol=1e-8, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    return contains_batch(pack(atlas), Pbatch; tol=tol, workspace=workspace)
end

function contains_batch(frozen::FrozenAtlas, Pbatch::AbstractMatrix; policy=frozen.policy, tol=policy.region_tol, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    return contains_batch(frozen.packed, Pbatch; tol=tol, workspace=workspace)
end

function query_batch(atlas::PackedRegionAtlas, Pbatch::AbstractMatrix; tol=1e-8, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    cert_ids = workspace === nothing ? zeros(Int, size(Pbatch, 2)) : _workspace_cert_ids(workspace, size(Pbatch, 2))
    vals = _packed_constraint_values(atlas, Pbatch; workspace)
    fill!(cert_ids, 0)
    for idx in 1:(length(atlas.row_offsets) - 1)
        all(!iszero, cert_ids) && break
        row_start = atlas.row_offsets[idx]
        row_stop = atlas.row_offsets[idx + 1] - 1
        ok = if row_stop < row_start
            trues(length(cert_ids))
        else
            vec(all(@view(vals[row_start:row_stop, :]) .>= -tol, dims=1))
        end
        for j in eachindex(cert_ids)
            cert_ids[j] == 0 && ok[j] && (cert_ids[j] = idx)
        end
    end
    return cert_ids
end

function _query_point(atlas::PackedRegionAtlas, p::AbstractVector; tol=1e-8, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    vals = workspace === nothing ? zeros(Float64, size(atlas.C, 1)) : _workspace_val_col(workspace, atlas)
    LinearAlgebra.mul!(vals, atlas.C, p)
    vals .+= atlas.c
    for idx in 1:(length(atlas.row_offsets) - 1)
        row_start = atlas.row_offsets[idx]
        row_stop = atlas.row_offsets[idx + 1] - 1
        feasible = true
        for row in row_start:row_stop
            if vals[row] < -tol
                feasible = false
                break
            end
        end
        feasible && return idx
    end
    return 0
end

function query_batch(atlas::RegionAtlas, Pbatch::AbstractMatrix; tol=1e-8, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    ids = query_batch(pack(atlas), Pbatch; tol=tol, workspace=workspace)
    _record_query_metrics!(atlas, ids)
    for idx in ids
        idx == 0 && continue
        _touch_certificate!(atlas, idx)
    end
    return ids
end

function query_batch(frozen::FrozenAtlas, Pbatch::AbstractMatrix; policy=frozen.policy, tol=policy.region_tol, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    ids = query_batch(frozen.packed, Pbatch; tol=tol, workspace=workspace)
    _record_query_metrics!(frozen, ids)
    return ids
end

function primal_batch(atlas::PackedRegionAtlas, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    length(cert_ids) == size(Pbatch, 2) || error("Certificate id vector must match batch size.")
    X = zeros(Float64, size(atlas.P, 1), size(Pbatch, 2))
    offsets, perm = _group_certificate_columns(cert_ids, size(atlas.P, 3); workspace)
    for idx in 1:size(atlas.P, 3)
        col_start = offsets[idx]
        col_stop = offsets[idx + 1] - 1
        col_stop < col_start && continue
        cols = @view perm[col_start:col_stop]
        LinearAlgebra.mul!(@view(X[:, cols]), @view(atlas.P[:, :, idx]), @view(Pbatch[:, cols]))
        @views X[:, cols] .+= atlas.p0[:, idx]
    end
    return X
end

function primal_batch(atlas::RegionAtlas, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    return primal_batch(pack(atlas), cert_ids, Pbatch; workspace=workspace)
end

function primal_batch(frozen::FrozenAtlas, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    return primal_batch(frozen.packed, cert_ids, Pbatch; workspace=workspace)
end

function dual_batch(atlas::PackedRegionAtlas, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    length(cert_ids) == size(Pbatch, 2) || error("Certificate id vector must match batch size.")
    Λ = zeros(Float64, size(atlas.D, 1), size(Pbatch, 2))
    offsets, perm = _group_certificate_columns(cert_ids, size(atlas.D, 3); workspace)
    for idx in 1:size(atlas.D, 3)
        col_start = offsets[idx]
        col_stop = offsets[idx + 1] - 1
        col_stop < col_start && continue
        cols = @view perm[col_start:col_stop]
        LinearAlgebra.mul!(@view(Λ[:, cols]), @view(atlas.D[:, :, idx]), @view(Pbatch[:, cols]))
        @views Λ[:, cols] .+= atlas.d0[:, idx]
    end
    return Λ
end

function dual_batch(atlas::RegionAtlas, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    return dual_batch(pack(atlas), cert_ids, Pbatch; workspace=workspace)
end

function dual_batch(frozen::FrozenAtlas, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    return dual_batch(frozen.packed, cert_ids, Pbatch; workspace=workspace)
end

function query(frozen::FrozenAtlas, p::AbstractVector; policy=frozen.policy, tol=policy.region_tol)
    idx = _query_point(frozen.packed, p; tol=tol)
    _metric_inc!(frozen.metrics, :query_calls)
    if idx == 0
        _metric_inc!(frozen.metrics, :query_misses)
        return nothing
    end
    _metric_inc!(frozen.metrics, :query_hits)
    return frozen.certificates[idx]
end

function contains_point(frozen::FrozenAtlas, p::AbstractVector; policy=frozen.policy, tol=policy.region_tol)
    return _query_point(frozen.packed, p; tol=tol) != 0
end

function clear!(atlas::RegionAtlas)
    atlas.frozen && throw(InvalidModelStateError("Cannot clear a frozen RegionAtlas."))
    empty!(atlas.certificates)
    empty!(atlas.certificate_rows)
    empty!(atlas.signatures)
    empty!(atlas.region_signatures)
    empty!(atlas.primal_signatures)
    empty!(atlas.cluster_members)
    empty!(atlas.certificate_clusters)
    empty!(atlas.certificate_representatives)
    empty!(atlas.constraint_store.C)
    empty!(atlas.constraint_store.c)
    empty!(atlas.constraint_store.refcount)
    empty!(atlas.constraint_store.index)
    atlas.last_hit = 0
    atlas.packed = nothing
    atlas.packed_dirty = true
    empty!(atlas.last_access)
    empty!(atlas.insertion_order)
    atlas.access_clock = 0
    reset_metrics!(atlas)
    return atlas
end

function remove_certificate!(atlas::RegionAtlas, idx::Integer)
    atlas.frozen && throw(InvalidModelStateError("Cannot mutate a frozen RegionAtlas."))
    1 <= idx <= length(atlas.certificates) || throw(BoundsError(atlas.certificates, idx))
    row_ids = atlas.certificate_rows[idx]
    for row_id in row_ids
        atlas.constraint_store.refcount[row_id] -= 1
    end
    deleteat!(atlas.certificates, idx)
    deleteat!(atlas.certificate_rows, idx)
    deleteat!(atlas.last_access, idx)
    deleteat!(atlas.insertion_order, idx)
    return _rebuild_atlas_metadata!(atlas)
end

function compact!(atlas::RegionAtlas)
    atlas.frozen && throw(InvalidModelStateError("Cannot compact a frozen RegionAtlas."))
    new_store = ConstraintStore()
    for rows in atlas.certificate_rows
        empty!(rows)
    end
    new_rows = Vector{Vector{Int}}(undef, length(atlas.certificates))
    for (i, cert) in enumerate(atlas.certificates)
        new_rows[i] = _store_constraints!(new_store, cert.C, cert.c; tol=atlas.policy.dedup_tol)
    end
    atlas.constraint_store = new_store
    atlas.certificate_rows = new_rows
    return _rebuild_atlas_metadata!(atlas)
end

function atlas_stats(atlas::RegionAtlas)
    n_cert = length(atlas.certificates)
    total_rows = sum(length, atlas.certificate_rows)
    shared_rows = length(atlas.constraint_store.c)
    packed_rows = atlas.packed === nothing ? total_rows : size(atlas.packed.C, 1)
    if n_cert == 0
        return AtlasStats(0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0)
    end
    cert0 = atlas.certificates[1]
    return AtlasStats(
        n_cert,
        length(atlas.cluster_members),
        length(atlas.region_signatures),
        total_rows,
        shared_rows,
        total_rows == 0 ? 1.0 : shared_rows / total_rows,
        size(cert0.C, 2),
        size(cert0.P, 1),
        size(cert0.D, 1),
        packed_rows,
    )
end

atlas_stats(frozen::FrozenAtlas) = frozen.stats

function freeze(atlas::RegionAtlas; force_repack=false)
    packed = pack(atlas; force=force_repack)
    return FrozenAtlas(copy(atlas.certificates), packed, atlas.policy, atlas.atlas_policy, atlas_stats(atlas), AtlasMetrics())
end

function thaw(frozen::FrozenAtlas)
    atlas = RegionAtlas(copy(frozen.certificates), 0; policy=frozen.policy, atlas_policy=frozen.atlas_policy, frozen=false)
    atlas.packed = frozen.packed
    atlas.packed_dirty = false
    return atlas
end

function snapshot(atlas::RegionAtlas)
    tmp = thaw(freeze(atlas))
    compact!(tmp)
    return AtlasSnapshot(
        1,
        Dates.format(Dates.now(), Dates.dateformat"yyyy-mm-ddTHH:MM:SS"),
        copy(tmp.certificates),
        copy(tmp.certificate_rows),
        copy(tmp.constraint_store.C),
        copy(tmp.constraint_store.c),
        copy(tmp.constraint_store.refcount),
        atlas.last_hit,
        tmp.policy,
        tmp.atlas_policy,
    )
end

snapshot(frozen::FrozenAtlas) = snapshot(thaw(frozen))

function contains_batch!(dest::AbstractMatrix{Bool}, atlas_like, Pbatch::AbstractMatrix; policy=default_policy(), tol=policy.region_tol, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    hits = contains_batch(atlas_like, Pbatch; tol=tol, workspace=workspace)
    size(dest) == size(hits) || throw(DimensionMismatch("Destination hit matrix has size $(size(dest)); expected $(size(hits))."))
    dest .= hits
    return dest
end

function query_batch!(dest::AbstractVector{Int}, atlas_like, Pbatch::AbstractMatrix; policy=default_policy(), tol=policy.region_tol, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    ids = query_batch(atlas_like, Pbatch; tol=tol, workspace=workspace)
    length(dest) == length(ids) || throw(DimensionMismatch("Destination id vector has length $(length(dest)); expected $(length(ids))."))
    dest .= ids
    return dest
end

function primal_batch!(dest::AbstractMatrix, atlas_like, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    X = primal_batch(atlas_like, cert_ids, Pbatch; workspace=workspace)
    size(dest) == size(X) || throw(DimensionMismatch("Destination primal matrix has size $(size(dest)); expected $(size(X))."))
    dest .= X
    return dest
end

function dual_batch!(dest::AbstractMatrix, atlas_like, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    Λ = dual_batch(atlas_like, cert_ids, Pbatch; workspace=workspace)
    size(dest) == size(Λ) || throw(DimensionMismatch("Destination dual matrix has size $(size(dest)); expected $(size(Λ))."))
    dest .= Λ
    return dest
end

function solve_batch(atlas_like, Pbatch::AbstractMatrix; policy=default_policy(), tol=policy.region_tol, workspace::Union{Nothing,PackedAtlasWorkspace}=nothing, require_coverage=true)
    ids = query_batch(atlas_like, Pbatch; tol=tol, workspace=workspace)
    if require_coverage && any(iszero, ids)
        missing = count(iszero, ids)
        throw(QueryCoverageError("Atlas does not cover $missing query point(s)."))
    end
    return BatchResult(ids, primal_batch(atlas_like, ids, Pbatch; workspace=workspace), dual_batch(atlas_like, ids, Pbatch; workspace=workspace))
end

function _group_certificate_columns(cert_ids::AbstractVector{Int}, n_cert::Int; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    counts = workspace === nothing ? zeros(Int, n_cert) : _workspace_counts(workspace, n_cert)
    offsets = workspace === nothing ? zeros(Int, n_cert + 1) : _workspace_offsets(workspace, n_cert)
    perm = workspace === nothing ? zeros(Int, length(cert_ids)) : _workspace_perm(workspace, length(cert_ids))

    fill!(counts, 0)
    for idx in cert_ids
        idx == 0 && continue
        1 <= idx <= n_cert || error("Certificate id $idx is out of bounds for packed atlas.")
        counts[idx] += 1
    end

    offsets[1] = 1
    for idx in 1:n_cert
        offsets[idx + 1] = offsets[idx] + counts[idx]
    end

    next_slot = counts
    for idx in 1:n_cert
        next_slot[idx] = offsets[idx]
    end
    for (j, idx) in pairs(cert_ids)
        idx == 0 && continue
        slot = next_slot[idx]
        perm[slot] = j
        next_slot[idx] += 1
    end
    return offsets, perm
end

function _packed_constraint_values(atlas::PackedRegionAtlas, Pbatch::AbstractMatrix; workspace::Union{Nothing,PackedAtlasWorkspace}=nothing)
    vals = workspace === nothing ? zeros(Float64, size(atlas.C, 1), size(Pbatch, 2)) : _workspace_vals(workspace, atlas, size(Pbatch, 2))
    LinearAlgebra.mul!(vals, atlas.C, Pbatch)
    vals .+= atlas.c
    return vals
end

function _workspace_vals(workspace::PackedAtlasWorkspace, atlas::PackedRegionAtlas, n_pts::Int)
    size(workspace.vals, 1) == size(atlas.C, 1) || error("Workspace row dimension does not match packed atlas.")
    size(workspace.vals, 2) >= n_pts || error("Workspace batch capacity is smaller than the requested batch.")
    return @view workspace.vals[:, 1:n_pts]
end

function _workspace_val_col(workspace::PackedAtlasWorkspace, atlas::PackedRegionAtlas)
    size(workspace.vals, 1) == size(atlas.C, 1) || error("Workspace row dimension does not match packed atlas.")
    size(workspace.vals, 2) >= 1 || error("Workspace batch capacity is smaller than the requested batch.")
    return @view workspace.vals[:, 1]
end

function _workspace_hits(workspace::PackedAtlasWorkspace, atlas::PackedRegionAtlas, n_pts::Int)
    size(workspace.hits, 1) == size(atlas.P, 3) || error("Workspace certificate dimension does not match packed atlas.")
    size(workspace.hits, 2) >= n_pts || error("Workspace batch capacity is smaller than the requested batch.")
    return @view workspace.hits[:, 1:n_pts]
end

function _workspace_cert_ids(workspace::PackedAtlasWorkspace, n_pts::Int)
    length(workspace.cert_ids) >= n_pts || error("Workspace batch capacity is smaller than the requested batch.")
    return @view workspace.cert_ids[1:n_pts]
end

function _workspace_counts(workspace::PackedAtlasWorkspace, n_cert::Int)
    length(workspace.counts) >= n_cert || error("Workspace certificate capacity is smaller than the packed atlas.")
    return @view workspace.counts[1:n_cert]
end

function _workspace_offsets(workspace::PackedAtlasWorkspace, n_cert::Int)
    length(workspace.offsets) >= n_cert + 1 || error("Workspace offset capacity is smaller than the packed atlas.")
    return @view workspace.offsets[1:(n_cert + 1)]
end

function _workspace_perm(workspace::PackedAtlasWorkspace, n_pts::Int)
    length(workspace.perm) >= n_pts || error("Workspace batch capacity is smaller than the requested batch.")
    return @view workspace.perm[1:n_pts]
end

function _certificate_signature(cert::BasisCertificate; tol=1e-8)
    basis_key = join(cert.basis_cols, ",")
    var_key = join(Int.(cert.var_basis), ",")
    row_key = join(Int.(cert.row_basis), ",")
    return string(basis_key, "|", var_key, "|", row_key, "|", _region_signature(cert; tol))
end

function _primal_cluster_signature(cert::BasisCertificate; tol=1e-8)
    pkey = join(vcat(vec(round.(Int, cert.P ./ tol)), round.(Int, cert.p0 ./ tol)), ",")
    return pkey
end

function _region_signature(cert::BasisCertificate; tol=1e-8)
    pkey = _primal_cluster_signature(cert; tol)
    row_keys = String[]
    for i in axes(cert.C, 1)
        row = cert.C[i, :]
        push!(row_keys, _constraint_row_key(row, cert.c[i]; tol))
    end
    sort!(row_keys)
    return string(pkey, "|", join(row_keys, ";"))
end

function _rebuild_atlas_metadata!(atlas::RegionAtlas)
    empty!(atlas.signatures)
    empty!(atlas.region_signatures)
    empty!(atlas.primal_signatures)
    empty!(atlas.cluster_members)
    empty!(atlas.certificate_clusters)
    empty!(atlas.certificate_representatives)
    for (idx, cert) in enumerate(atlas.certificates)
        sig = _certificate_signature(cert; tol=atlas.policy.dedup_tol)
        region_sig = _region_signature(cert; tol=atlas.policy.dedup_tol)
        cluster_sig = _primal_cluster_signature(cert; tol=atlas.policy.dedup_tol)
        atlas.signatures[sig] = idx
        atlas.region_signatures[region_sig] = idx
        cluster_id = get!(atlas.primal_signatures, cluster_sig) do
            push!(atlas.cluster_members, Int[])
            return length(atlas.cluster_members)
        end
        push!(atlas.certificate_clusters, cluster_id)
        push!(atlas.certificate_representatives, idx)
        push!(atlas.cluster_members[cluster_id], idx)
    end
    if length(atlas.last_access) != length(atlas.certificates)
        atlas.last_access = collect(1:length(atlas.certificates))
    end
    if length(atlas.insertion_order) != length(atlas.certificates)
        atlas.insertion_order = collect(1:length(atlas.certificates))
    end
    atlas.access_clock = isempty(atlas.last_access) ? 0 : max(maximum(atlas.last_access), maximum(atlas.insertion_order))
    atlas.last_hit = isempty(atlas.certificates) ? 0 : min(atlas.last_hit, length(atlas.certificates))
    atlas.packed = nothing
    atlas.packed_dirty = true
    return atlas
end

function _store_constraints!(store::ConstraintStore, C::AbstractMatrix, c::AbstractVector; tol=1e-8)
    row_ids = Int[]
    for i in axes(C, 1)
        row = collect(@view C[i, :])
        key = _constraint_row_key(row, c[i]; tol)
        row_id = get!(store.index, key) do
            push!(store.C, row)
            push!(store.c, c[i])
            push!(store.refcount, 0)
            return length(store.c)
        end
        store.refcount[row_id] += 1
        push!(row_ids, row_id)
    end
    return row_ids
end

function _constraint_row_key(row::AbstractVector, rhs; tol=1e-8)
    vals = vcat(collect(row), [rhs])
    return join(round.(Int, vals ./ tol), ",")
end

function _load_constraint_rows!(C::Matrix{Float64}, c::Vector{Float64}, row_range, store::ConstraintStore, row_ids::Vector{Int})
    for (local_idx, row_id) in enumerate(row_ids)
        row = store.C[row_id]
        global_idx = first(row_range) + local_idx - 1
        C[global_idx, :] .= row
        c[global_idx] = store.c[row_id]
    end
    return nothing
end
