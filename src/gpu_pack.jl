function gpu_pack(atlas_like; T::Type{<:AbstractFloat}=Float32, I::Type{<:Integer}=Int32)
    packed = pack(atlas_like)
    return GPUPackedRegionAtlas{T,I}(
        T.(packed.C),
        T.(packed.c),
        I.(packed.row_offsets),
        T.(packed.P),
        T.(packed.p0),
        T.(packed.D),
        T.(packed.d0),
    )
end

function contains_batch(atlas::GPUPackedRegionAtlas, Pbatch::AbstractMatrix; tol=1e-6)
    vals = atlas.C * _gpu_promote_batch(atlas, Pbatch)
    vals .+= atlas.c
    n_cert = length(atlas.row_offsets) - 1
    hits = falses(n_cert, size(Pbatch, 2))
    tolT = eltype(atlas.C)(tol)
    for idx in 1:n_cert
        row_start = Int(atlas.row_offsets[idx])
        row_stop = Int(atlas.row_offsets[idx + 1]) - 1
        if row_stop < row_start
            hits[idx, :] .= true
            continue
        end
        hits[idx, :] .= vec(all(@view(vals[row_start:row_stop, :]) .>= -tolT, dims=1))
    end
    return hits
end

function query_batch(atlas::GPUPackedRegionAtlas, Pbatch::AbstractMatrix; tol=1e-6)
    vals = atlas.C * _gpu_promote_batch(atlas, Pbatch)
    vals .+= atlas.c
    cert_ids = zeros(Int, size(Pbatch, 2))
    tolT = eltype(atlas.C)(tol)
    for idx in 1:(length(atlas.row_offsets) - 1)
        all(!iszero, cert_ids) && break
        row_start = Int(atlas.row_offsets[idx])
        row_stop = Int(atlas.row_offsets[idx + 1]) - 1
        ok = if row_stop < row_start
            trues(length(cert_ids))
        else
            vec(all(@view(vals[row_start:row_stop, :]) .>= -tolT, dims=1))
        end
        for j in eachindex(cert_ids)
            cert_ids[j] == 0 && ok[j] && (cert_ids[j] = idx)
        end
    end
    return cert_ids
end

function primal_batch(atlas::GPUPackedRegionAtlas, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix)
    length(cert_ids) == size(Pbatch, 2) || error("Certificate id vector must match batch size.")
    X = zeros(eltype(atlas.P), size(atlas.P, 1), size(Pbatch, 2))
    PbatchT = _gpu_promote_batch(atlas, Pbatch)
    for idx in 1:size(atlas.P, 3)
        cols = findall(==(idx), cert_ids)
        isempty(cols) && continue
        @views X[:, cols] .= atlas.P[:, :, idx] * PbatchT[:, cols]
        @views X[:, cols] .+= atlas.p0[:, idx]
    end
    return X
end

function dual_batch(atlas::GPUPackedRegionAtlas, cert_ids::AbstractVector{Int}, Pbatch::AbstractMatrix)
    length(cert_ids) == size(Pbatch, 2) || error("Certificate id vector must match batch size.")
    Λ = zeros(eltype(atlas.D), size(atlas.D, 1), size(Pbatch, 2))
    PbatchT = _gpu_promote_batch(atlas, Pbatch)
    for idx in 1:size(atlas.D, 3)
        cols = findall(==(idx), cert_ids)
        isempty(cols) && continue
        @views Λ[:, cols] .= atlas.D[:, :, idx] * PbatchT[:, cols]
        @views Λ[:, cols] .+= atlas.d0[:, idx]
    end
    return Λ
end

function solve_batch(atlas::GPUPackedRegionAtlas, Pbatch::AbstractMatrix; tol=1e-6, require_coverage=true)
    ids = query_batch(atlas, Pbatch; tol)
    if require_coverage && any(iszero, ids)
        missing = count(iszero, ids)
        throw(QueryCoverageError("Atlas does not cover $missing query point(s)."))
    end
    return BatchResult(ids, Matrix(primal_batch(atlas, ids, Pbatch)), Matrix(dual_batch(atlas, ids, Pbatch)))
end

function _gpu_promote_batch(atlas::GPUPackedRegionAtlas{T}, Pbatch::AbstractMatrix) where {T}
    if eltype(Pbatch) === T
        return Pbatch
    end
    return T.(Pbatch)
end
