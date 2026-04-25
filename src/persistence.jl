const _ATLAS_ARCHIVE_VERSION = 3

function save_atlas(path::AbstractString, atlas_like)
    snap = snapshot(atlas_like)
    mktempdir() do tmpdir
        root = joinpath(tmpdir, "atlas")
        mkpath(root)
        manifest = _write_stable_snapshot!(root, snap)
        open(joinpath(root, "manifest.toml"), "w") do io
            TOML.print(io, manifest)
        end
        Tar.create(root, path)
    end
    return path
end

function load_atlas(path::AbstractString; frozen=false)
    snap = _load_snapshot(path)
    atlas = RegionAtlas(; policy=snap.policy, atlas_policy=snap.atlas_policy, frozen=false)
    atlas.certificates = copy(snap.certificates)
    atlas.certificate_rows = copy(snap.certificate_rows)
    atlas.constraint_store = ConstraintStore(copy(snap.constraint_C), copy(snap.constraint_c), copy(snap.constraint_refcount), Dict{String,Int}())
    for i in eachindex(atlas.constraint_store.c)
        atlas.constraint_store.index[_constraint_row_key(atlas.constraint_store.C[i], atlas.constraint_store.c[i]; tol=atlas.policy.dedup_tol)] = i
    end
    atlas.last_hit = isempty(atlas.certificates) ? 0 : min(snap.last_hit, length(atlas.certificates))
    _rebuild_atlas_metadata!(atlas)
    return frozen ? freeze(atlas) : atlas
end

function _load_snapshot(path::AbstractString)
    try
        return _load_stable_snapshot(path)
    catch err
        err isa InterruptException && rethrow()
        try
            snap = Serialization.deserialize(path)::AtlasSnapshot
            snap.format_version in (1, 2) || throw(InvalidModelStateError("Unsupported atlas snapshot format version $(snap.format_version)."))
            return snap
        catch legacy_err
            legacy_err isa InterruptException && rethrow()
            throw(InvalidModelStateError("Unable to load atlas from $path. Stable archive load failed with $(typeof(err)); legacy snapshot load failed with $(typeof(legacy_err))."))
        end
    end
end

function _load_stable_snapshot(path::AbstractString)
    return mktempdir() do tmpdir
        Tar.extract(path, tmpdir)
        root = _stable_archive_root(tmpdir)
        manifest = TOML.parsefile(joinpath(root, "manifest.toml"))
        version = Int(manifest["format_version"])
        version in (2, _ATLAS_ARCHIVE_VERSION) || throw(InvalidModelStateError("Unsupported stable atlas archive format version $version."))
        policy = _decode_policy(manifest["policy"])
        atlas_policy = version >= 3 && haskey(manifest, "atlas_policy") ? _decode_atlas_policy(manifest["atlas_policy"]) : default_atlas_policy()
        certificates = BasisCertificate[]
        cert_rows = Vector{Vector{Int}}()
        for cert_meta in manifest["certificates"]
            push!(certificates, _read_certificate(root, cert_meta))
        end
        for rows_meta in manifest["certificate_rows"]
            push!(cert_rows, _read_int_vector(root, rows_meta))
        end
        constraint_C = Vector{Vector{Float64}}()
        for row_meta in manifest["constraint_store"]["rows"]
            push!(constraint_C, _read_float_vector(root, row_meta))
        end
        constraint_c = _read_float_vector(root, manifest["constraint_store"]["rhs"])
        constraint_refcount = _read_int_vector(root, manifest["constraint_store"]["refcount"])
        return AtlasSnapshot(
            2,
            String(manifest["created_at"]),
            certificates,
            cert_rows,
            constraint_C,
            constraint_c,
            constraint_refcount,
            Int(manifest["last_hit"]),
            policy,
            atlas_policy,
        )
    end
end

function _write_stable_snapshot!(root::AbstractString, snap::AtlasSnapshot)
    constraints_dir = joinpath(root, "constraints")
    cert_dir = joinpath(root, "certificates")
    rows_dir = joinpath(root, "certificate_rows")
    mkpath(constraints_dir)
    mkpath(cert_dir)
    mkpath(rows_dir)

    constraint_rows_meta = Any[]
    for (i, row) in enumerate(snap.constraint_C)
        rel = joinpath("constraints", "row_$i.bin")
        push!(constraint_rows_meta, _write_float_vector(joinpath(root, rel), row; relpath=rel))
    end
    constraint_rhs_meta = _write_float_vector(joinpath(root, "constraints", "rhs.bin"), snap.constraint_c; relpath=joinpath("constraints", "rhs.bin"))
    constraint_refcount_meta = _write_int_vector(joinpath(root, "constraints", "refcount.bin"), snap.constraint_refcount; relpath=joinpath("constraints", "refcount.bin"))

    certs_meta = Any[]
    for (i, cert) in enumerate(snap.certificates)
        cert_root = joinpath(cert_dir, "cert_$i")
        mkpath(cert_root)
        push!(certs_meta, _write_certificate!(root, cert_root, cert))
    end

    rows_meta = Any[]
    for (i, rows) in enumerate(snap.certificate_rows)
        rel = joinpath("certificate_rows", "rows_$i.bin")
        push!(rows_meta, _write_int_vector(joinpath(root, rel), rows; relpath=rel))
    end

    return Dict(
        "format_version" => _ATLAS_ARCHIVE_VERSION,
        "created_at" => snap.created_at,
        "last_hit" => snap.last_hit,
        "policy" => _encode_policy(snap.policy),
        "atlas_policy" => _encode_atlas_policy(snap.atlas_policy),
        "certificates" => certs_meta,
        "certificate_rows" => rows_meta,
        "constraint_store" => Dict(
            "rows" => constraint_rows_meta,
            "rhs" => constraint_rhs_meta,
            "refcount" => constraint_refcount_meta,
        ),
    )
end

function _write_certificate!(archive_root::AbstractString, root::AbstractString, cert::BasisCertificate)
    return Dict(
        "basis_cols" => _write_int_vector(joinpath(root, "basis_cols.bin"), cert.basis_cols; relpath=relpath(joinpath(root, "basis_cols.bin"), archive_root)),
        "anchor_p" => _write_float_vector(joinpath(root, "anchor_p.bin"), cert.anchor_p; relpath=relpath(joinpath(root, "anchor_p.bin"), archive_root)),
        "C" => _write_float_matrix(joinpath(root, "region_matrix.bin"), cert.C; relpath=relpath(joinpath(root, "region_matrix.bin"), archive_root)),
        "c" => _write_float_vector(joinpath(root, "region_rhs.bin"), cert.c; relpath=relpath(joinpath(root, "region_rhs.bin"), archive_root)),
        "P" => _write_float_matrix(joinpath(root, "P.bin"), cert.P; relpath=relpath(joinpath(root, "P.bin"), archive_root)),
        "p0" => _write_float_vector(joinpath(root, "p0.bin"), cert.p0; relpath=relpath(joinpath(root, "p0.bin"), archive_root)),
        "D" => _write_float_matrix(joinpath(root, "D.bin"), cert.D; relpath=relpath(joinpath(root, "D.bin"), archive_root)),
        "d0" => _write_float_vector(joinpath(root, "d0.bin"), cert.d0; relpath=relpath(joinpath(root, "d0.bin"), archive_root)),
        "var_basis" => _write_basis_status_vector(joinpath(root, "var_basis.bin"), cert.var_basis; relpath=relpath(joinpath(root, "var_basis.bin"), archive_root)),
        "row_basis" => _write_basis_status_vector(joinpath(root, "row_basis.bin"), cert.row_basis; relpath=relpath(joinpath(root, "row_basis.bin"), archive_root)),
        "basis_hint" => _write_basis_hint!(archive_root, root, cert.basis_hint),
    )
end

function _read_certificate(root::AbstractString, meta)
    basis_cols = _read_int_vector(root, meta["basis_cols"])
    anchor_p = _read_float_vector(root, meta["anchor_p"])
    C = _read_float_matrix(root, meta["C"])
    c = _read_float_vector(root, meta["c"])
    P = _read_float_matrix(root, meta["P"])
    p0 = _read_float_vector(root, meta["p0"])
    D = _read_float_matrix(root, meta["D"])
    d0 = _read_float_vector(root, meta["d0"])
    var_basis = _read_basis_status_vector(root, meta["var_basis"])
    row_basis = _read_basis_status_vector(root, meta["row_basis"])
    basis_hint = _read_basis_hint(root, meta["basis_hint"])
    return BasisCertificate(basis_cols, anchor_p, C, c, P, p0, D, d0, var_basis, row_basis, basis_hint)
end

function _write_basis_hint!(archive_root::AbstractString, root::AbstractString, hint::NoBasisHint)
    return Dict("kind" => "none")
end

function _write_basis_hint!(archive_root::AbstractString, root::AbstractString, hint::HiGHSBasisHint)
    return Dict(
        "kind" => "highs",
        "col_status" => _write_int32_vector(joinpath(root, "hint_col_status.bin"), Int32.(hint.col_status); relpath=relpath(joinpath(root, "hint_col_status.bin"), archive_root)),
        "row_status" => _write_int32_vector(joinpath(root, "hint_row_status.bin"), Int32.(hint.row_status); relpath=relpath(joinpath(root, "hint_row_status.bin"), archive_root)),
    )
end

function _write_basis_hint!(archive_root::AbstractString, root::AbstractString, hint::MOIBasisHint)
    return Dict(
        "kind" => "moi",
        "var_status" => _write_basis_status_vector(joinpath(root, "hint_var_status.bin"), hint.var_status; relpath=relpath(joinpath(root, "hint_var_status.bin"), archive_root)),
        "row_status" => _write_basis_status_vector(joinpath(root, "hint_row_status.bin"), hint.row_status; relpath=relpath(joinpath(root, "hint_row_status.bin"), archive_root)),
    )
end

function _read_basis_hint(root::AbstractString, meta)
    kind = String(meta["kind"])
    if kind == "none"
        return NoBasisHint()
    elseif kind == "highs"
        return HiGHSBasisHint(HiGHS.HighsInt.(_read_int32_vector(root, meta["col_status"])), HiGHS.HighsInt.(_read_int32_vector(root, meta["row_status"])))
    elseif kind == "moi"
        return MOIBasisHint(_read_basis_status_vector(root, meta["var_status"]), _read_basis_status_vector(root, meta["row_status"]))
    end
    throw(InvalidModelStateError("Unknown persisted basis hint kind $kind."))
end

_encode_policy(policy::NumericalPolicy) = Dict(
    "zero_tol" => policy.zero_tol,
    "basis_tol" => policy.basis_tol,
    "region_tol" => policy.region_tol,
    "reduction_tol" => policy.reduction_tol,
    "dedup_tol" => policy.dedup_tol,
    "exact_reduction" => policy.exact_reduction,
    "exact_reduction_param_limit" => policy.exact_reduction_param_limit,
    "exact_reduction_row_limit" => policy.exact_reduction_row_limit,
)

_decode_policy(meta) = NumericalPolicy(
    Float64(meta["zero_tol"]),
    Float64(meta["basis_tol"]),
    Float64(meta["region_tol"]),
    Float64(meta["reduction_tol"]),
    Float64(meta["dedup_tol"]),
    Bool(meta["exact_reduction"]),
    Int(meta["exact_reduction_param_limit"]),
    Int(meta["exact_reduction_row_limit"]),
)

_encode_atlas_policy(policy::AtlasPolicy) = Dict(
    "max_certificates" => policy.max_certificates,
    "max_shared_rows" => policy.max_shared_rows,
    "learning_enabled" => policy.learning_enabled,
    "eviction" => String(policy.eviction),
)

_decode_atlas_policy(meta) = AtlasPolicy(
    max_certificates=Int(meta["max_certificates"]),
    max_shared_rows=Int(meta["max_shared_rows"]),
    learning_enabled=Bool(meta["learning_enabled"]),
    eviction=Symbol(meta["eviction"]),
)

function _write_float_vector(path::AbstractString, v::AbstractVector{<:Real}; relpath=basename(path))
    data = Vector{Float64}(v)
    open(path, "w") do io
        write(io, reinterpret(UInt8, data))
    end
    return Dict("path" => relpath, "length" => length(data), "kind" => "float64_vector")
end

function _read_float_vector(root::AbstractString, meta)
    path = joinpath(root, String(meta["path"]))
    len = Int(meta["length"])
    data = Vector{Float64}(undef, len)
    open(path, "r") do io
        read!(io, reinterpret(UInt8, data))
    end
    return data
end

function _write_float_matrix(path::AbstractString, A::AbstractMatrix{<:Real}; relpath=basename(path))
    data = Matrix{Float64}(A)
    open(path, "w") do io
        write(io, reinterpret(UInt8, vec(data)))
    end
    return Dict("path" => relpath, "size" => [size(data, 1), size(data, 2)], "kind" => "float64_matrix")
end

function _read_float_matrix(root::AbstractString, meta)
    dims = Tuple(Int.(meta["size"]))
    data = Matrix{Float64}(undef, dims...)
    open(joinpath(root, String(meta["path"])), "r") do io
        read!(io, reinterpret(UInt8, vec(data)))
    end
    return data
end

function _write_int_vector(path::AbstractString, v::AbstractVector{<:Integer}; relpath=basename(path))
    data = Vector{Int64}(v)
    open(path, "w") do io
        write(io, reinterpret(UInt8, data))
    end
    return Dict("path" => relpath, "length" => length(data), "kind" => "int64_vector")
end

function _read_int_vector(root::AbstractString, meta)
    len = Int(meta["length"])
    data = Vector{Int64}(undef, len)
    open(joinpath(root, String(meta["path"])), "r") do io
        read!(io, reinterpret(UInt8, data))
    end
    return Int.(data)
end

function _write_int32_vector(path::AbstractString, v::AbstractVector{<:Integer}; relpath=basename(path))
    data = Vector{Int32}(v)
    open(path, "w") do io
        write(io, reinterpret(UInt8, data))
    end
    return Dict("path" => relpath, "length" => length(data), "kind" => "int32_vector")
end

function _read_int32_vector(root::AbstractString, meta)
    len = Int(meta["length"])
    data = Vector{Int32}(undef, len)
    open(joinpath(root, String(meta["path"])), "r") do io
        read!(io, reinterpret(UInt8, data))
    end
    return data
end

function _write_basis_status_vector(path::AbstractString, v::AbstractVector{MOI.BasisStatusCode}; relpath=basename(path))
    return _write_int32_vector(path, Int32.(Int.(v)); relpath=relpath)
end

function _read_basis_status_vector(root::AbstractString, meta)
    return MOI.BasisStatusCode.(_read_int32_vector(root, meta))
end

function _stable_archive_root(tmpdir::AbstractString)
    nested = joinpath(tmpdir, "atlas")
    isfile(joinpath(nested, "manifest.toml")) && return nested
    isfile(joinpath(tmpdir, "manifest.toml")) && return tmpdir
    throw(InvalidModelStateError("Stable atlas archive is missing manifest.toml."))
end
