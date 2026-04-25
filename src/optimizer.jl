mutable struct Optimizer{OT} <: MOI.AbstractOptimizer
    optimizer_constructor::Type{OT}
    inner::DiffOpt.Optimizer
    atlas::RegionAtlas
    model::Union{Nothing,JuMP.Model}
    lookup_workspace::Union{Nothing,PackedAtlasWorkspace}
    param_buffer::Vector{Float64}
    use_cached::Bool
    cached_primal::Vector{Float64}
    cached_dual::Vector{Float64}
    cached_primal_pos::Vector{Int}
    cached_dual_pos::Vector{Int}
    primal_eval_buffer::Vector{Float64}
    dual_eval_buffer::Vector{Float64}
    cached_obj_val::Float64
    use_basis_hints::Bool
    basis_hint_attempts::Int
    basis_hint_applied::Int
end

const _PACKED_LOOKUP_THRESHOLD = 4
const _FULL_HINT_SCAN_LIMIT = 64

function Optimizer{OT}() where {OT}
    test_opt = OT()
    _native_basis_backend(test_opt) === nothing &&
        error("AffineRegions currently supports solvers with native basis-status access only. Implemented backends: HiGHS and optional extensions.")
    inner = DiffOpt.diff_optimizer(OT)
    MOI.set(inner, DiffOpt.ModelConstructor(), DiffOpt.BasisLinearProgram.GeneralModel)
    return Optimizer{OT}(
        OT,
        inner,
        RegionAtlas(),
        nothing,
        nothing,
        Float64[],
        false,
        Float64[],
        Float64[],
        Int[],
        Int[],
        Float64[],
        Float64[],
        0.0,
        false,
        0,
        0,
    )
end

function Model(optimizer_constructor; silent=true, policy=default_policy(), atlas_policy=default_atlas_policy(), use_basis_hints=false)
    opt = Optimizer{optimizer_constructor}()
    opt.atlas.policy = policy
    opt.atlas.atlas_policy = atlas_policy
    opt.use_basis_hints = use_basis_hints
    model = JuMP.direct_model(opt)
    opt.model = model
    silent && JuMP.set_silent(model)
    JuMP.set_optimize_hook(model, _optimize_hook)
    return model
end

function _optimize_hook(model; kwargs...)
    opt = JuMP.backend(model)::Optimizer
    opt.use_cached = false

    cert, p = _find_certificate(opt, model)
    if cert !== nothing
        _cache_solution!(opt, cert, model, p)
        opt.use_cached = true
        return
    end

    _apply_basis_hint!(opt, model, p)
    JuMP.optimize!(model; ignore_optimize_hook=true, kwargs...)
    if JuMP.is_solved_and_feasible(model)
        cert = certificate(model; policy=opt.atlas.policy)
        add_certificate!(opt.atlas, cert)
    end
end

function _find_certificate(opt::Optimizer, model::JuMP.Model; tol=opt.atlas.policy.region_tol)
    atlas = opt.atlas
    p = _current_param_vector!(opt, model)
    if atlas.last_hit > 0 && atlas.last_hit <= length(atlas.certificates)
        cert = atlas.certificates[atlas.last_hit]
        contains_point(cert, p; tol) && return cert, p
    end
    if length(atlas.certificates) >= _PACKED_LOOKUP_THRESHOLD
        packed = pack(atlas)
        workspace = _lookup_workspace!(opt, packed)
        idx = _query_point(packed, p; tol=tol, workspace=workspace)
        if idx != 0
            atlas.last_hit = idx
            return atlas.certificates[idx], p
        end
        return nothing, p
    end
    for (i, cert) in enumerate(atlas.certificates)
        i == atlas.last_hit && continue
        if contains_point(cert, p; tol)
            atlas.last_hit = i
            return cert, p
        end
    end
    return nothing, p
end

function _apply_basis_hint!(opt::Optimizer, model::JuMP.Model, p::AbstractVector; tol=opt.atlas.policy.region_tol)
    opt.use_basis_hints || return
    lp = extract_lp(model)
    length(lp.variables) <= 1 && return
    cert = _best_basis_hint(opt.atlas, p; tol)
    cert === nothing && return
    opt.basis_hint_attempts += 1
    applied = _set_basis_hint!(JuMP.backend(model), lp.variables, _row_constraints(model), cert)
    applied && (opt.basis_hint_applied += 1)
    return
end

function _best_basis_hint(atlas::RegionAtlas, p::AbstractVector; tol=1e-8)
    atlas.last_hit > 0 || return nothing
    atlas.last_hit <= length(atlas.certificates) || return nothing
    candidates = _hint_candidate_ids(atlas)
    best_idx = first(candidates)
    best_score = _basis_hint_score(atlas.certificates[best_idx], p; tol)
    for idx in candidates
        idx == best_idx && continue
        score = _basis_hint_score(atlas.certificates[idx], p; tol)
        if score < best_score
            best_idx = idx
            best_score = score
        end
    end
    return atlas.certificates[best_idx]
end

function _hint_candidate_ids(atlas::RegionAtlas)
    n_cert = length(atlas.certificates)
    if n_cert <= _FULL_HINT_SCAN_LIMIT
        return collect(1:n_cert)
    end
    lo = max(1, atlas.last_hit - 4)
    hi = min(n_cert, atlas.last_hit + 4)
    cluster_id = atlas.certificate_clusters[atlas.last_hit]
    candidates = Int[]
    for idx in lo:hi
        push!(candidates, idx)
    end
    for idx in atlas.cluster_members[cluster_id]
        idx in candidates || push!(candidates, idx)
    end
    return candidates
end

function _basis_hint_score(cert::BasisCertificate, p::AbstractVector; tol=1e-8)
    anchor_score = 0.0
    for i in eachindex(p)
        δ = p[i] - cert.anchor_p[i]
        anchor_score += δ * δ
    end
    return anchor_score + _region_violation_score(cert, p; tol)
end

function _region_violation_score(cert::BasisCertificate, p::AbstractVector; tol=1e-8)
    vals = cert.C * p
    vals .+= cert.c
    score = 0.0
    for v in vals
        if v < -tol
            score -= v
        end
    end
    return score
end

function _set_basis_hint!(backend, variables, row_constraints, cert::BasisCertificate)
    native = _native_basis_backend(backend)
    native === nothing && return false
    return _apply_native_basis_hint!(native, backend, variables, row_constraints, cert.basis_hint)
end

function _cache_solution!(opt::Optimizer, cert::BasisCertificate, model::JuMP.Model, p::AbstractVector)
    lp = extract_lp(model)
    all_vars = get!(model.ext, :_AffineRegions_jl_AllVariables) do
        JuMP.all_variables(model)
    end
    variables, parameters, duals = get!(model.ext, :_AffineRegions_jl_VarClass) do
        _classify_model_entities(model)
    end
    primal_map = _cached_primal_positions(model, all_vars)
    dual_map = _cached_dual_positions(model, duals)
    opt.cached_primal_pos = primal_map
    opt.cached_dual_pos = dual_map

    length(opt.cached_primal) == length(all_vars) || resize!(opt.cached_primal, length(all_vars))
    length(opt.cached_dual) == length(duals) || resize!(opt.cached_dual, length(duals))
    length(opt.primal_eval_buffer) == length(variables) || resize!(opt.primal_eval_buffer, length(variables))
    length(opt.dual_eval_buffer) == length(duals) || resize!(opt.dual_eval_buffer, length(duals))

    for (j, vr) in enumerate(parameters)
        opt.cached_primal[primal_map[JuMP.index(vr).value]] = p[j]
    end

    x = opt.primal_eval_buffer
    LinearAlgebra.mul!(x, cert.P, p)
    x .+= cert.p0
    for (i, vr) in enumerate(variables)
        opt.cached_primal[primal_map[JuMP.index(vr).value]] = x[i]
    end

    λ = opt.dual_eval_buffer
    LinearAlgebra.mul!(λ, cert.D, p)
    λ .+= cert.d0
    for (i, cr) in enumerate(duals)
        opt.cached_dual[dual_map[JuMP.index(cr).value]] = λ[i]
    end

    opt.cached_obj_val = _objective_value(lp, x, p)
end

function _current_param_vector!(opt::Optimizer, model::JuMP.Model)
    parameters = extract_lp(model).parameters
    return _get_param_vector!(opt.param_buffer, parameters)
end

function _lookup_workspace!(opt::Optimizer, packed::PackedRegionAtlas)
    if opt.lookup_workspace === nothing ||
       size(opt.lookup_workspace.vals, 1) != size(packed.C, 1) ||
       size(opt.lookup_workspace.hits, 1) != size(packed.P, 3)
        opt.lookup_workspace = PackedAtlasWorkspace(packed, 1)
    end
    return opt.lookup_workspace
end

function _cached_primal_positions(model::JuMP.Model, all_vars=JuMP.all_variables(model))
    return get!(model.ext, :_AffineRegions_jl_CachedPrimalPositions) do
        pos = zeros(Int, maximum((JuMP.index(vr).value for vr in all_vars); init=0))
        for (i, vr) in enumerate(all_vars)
            pos[JuMP.index(vr).value] = i
        end
        pos
    end
end

function _cached_dual_positions(model::JuMP.Model, duals)
    return get!(model.ext, :_AffineRegions_jl_CachedDualPositions) do
        pos = zeros(Int, maximum((JuMP.index(cr).value for cr in duals); init=0))
        for (i, cr) in enumerate(duals)
            pos[JuMP.index(cr).value] = i
        end
        pos
    end
end

MOI.add_variable(s::Optimizer) = MOI.add_variable(s.inner)
MOI.add_variables(s::Optimizer, n::Int) = MOI.add_variables(s.inner, n)
MOI.add_constraint(s::Optimizer, f::MOI.AbstractFunction, set::MOI.AbstractSet) = MOI.add_constraint(s.inner, f, set)
MOI.supports_constraint(s::Optimizer, ::Type{F}, ::Type{S}) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet} =
    MOI.supports_constraint(s.inner, F, S)
MOI.delete(s::Optimizer, vi::MOI.VariableIndex) = MOI.delete(s.inner, vi)
MOI.delete(s::Optimizer, ci::MOI.ConstraintIndex) = MOI.delete(s.inner, ci)
MOI.delete(s::Optimizer, vis::Vector{MOI.VariableIndex}) = MOI.delete(s.inner, vis)
MOI.is_valid(s::Optimizer, vi::MOI.VariableIndex) = MOI.is_valid(s.inner, vi)
MOI.is_valid(s::Optimizer, ci::MOI.ConstraintIndex) = MOI.is_valid(s.inner, ci)

function MOI.empty!(s::Optimizer)
    MOI.empty!(s.inner)
    clear!(s.atlas)
    s.lookup_workspace = nothing
    empty!(s.param_buffer)
    s.use_cached = false
    empty!(s.cached_primal)
    empty!(s.cached_dual)
    empty!(s.cached_primal_pos)
    empty!(s.cached_dual_pos)
    empty!(s.primal_eval_buffer)
    empty!(s.dual_eval_buffer)
    s.cached_obj_val = 0.0
    s.use_basis_hints = true
    s.basis_hint_attempts = 0
    s.basis_hint_applied = 0
end

MOI.is_empty(s::Optimizer) = MOI.is_empty(s.inner)
MOI.supports_incremental_interface(s::Optimizer) = MOI.supports_incremental_interface(s.inner)
MOI.copy_to(s::Optimizer, src::MOI.ModelLike) = MOI.copy_to(s.inner, src)
MOI.modify(s::Optimizer, ci::MOI.ConstraintIndex, chg::MOI.AbstractFunctionModification) = MOI.modify(s.inner, ci, chg)
MOI.modify(s::Optimizer, attr::MOI.ObjectiveFunction, chg::MOI.AbstractFunctionModification) = MOI.modify(s.inner, attr, chg)
MOI.get(s::Optimizer, attr::MOI.AbstractModelAttribute) = MOI.get(s.inner, attr)
MOI.set(s::Optimizer, attr::MOI.AbstractModelAttribute, value) = MOI.set(s.inner, attr, value)
MOI.supports(s::Optimizer, attr::MOI.AbstractModelAttribute) = MOI.supports(s.inner, attr)
MOI.get(s::Optimizer, attr::MOI.AbstractOptimizerAttribute) = MOI.get(s.inner, attr)
MOI.set(s::Optimizer, attr::MOI.AbstractOptimizerAttribute, value) = MOI.set(s.inner, attr, value)
MOI.supports(s::Optimizer, attr::MOI.AbstractOptimizerAttribute) = MOI.supports(s.inner, attr)
MOI.get(s::Optimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex) = MOI.get(s.inner, attr, vi)
MOI.set(s::Optimizer, attr::MOI.AbstractVariableAttribute, vi::MOI.VariableIndex, value) = MOI.set(s.inner, attr, vi, value)
MOI.supports(s::Optimizer, attr::MOI.AbstractVariableAttribute, ::Type{MOI.VariableIndex}) = MOI.supports(s.inner, attr, MOI.VariableIndex)
MOI.get(s::Optimizer, attr::MOI.AbstractConstraintAttribute, ci::MOI.ConstraintIndex) = MOI.get(s.inner, attr, ci)
MOI.set(s::Optimizer, attr::MOI.AbstractConstraintAttribute, ci::MOI.ConstraintIndex, value) = MOI.set(s.inner, attr, ci, value)
MOI.supports(s::Optimizer, attr::MOI.AbstractConstraintAttribute, ::Type{MOI.ConstraintIndex{F,S}}) where {F,S} =
    MOI.supports(s.inner, attr, MOI.ConstraintIndex{F,S})
MOI.get(s::Optimizer, ::Type{MOI.VariableIndex}, name::String) = MOI.get(s.inner, MOI.VariableIndex, name)
MOI.get(s::Optimizer, ::Type{MOI.ConstraintIndex}, name::String) = MOI.get(s.inner, MOI.ConstraintIndex, name)
MOI.get(s::Optimizer, ::Type{MOI.ConstraintIndex{F,S}}, name::String) where {F,S} = MOI.get(s.inner, MOI.ConstraintIndex{F,S}, name)
MOI.get(s::Optimizer, attr::MOI.ListOfConstraintIndices{F,S}) where {F,S} = MOI.get(s.inner, attr)

function MOI.get(s::Optimizer, ::MOI.TerminationStatus)
    s.use_cached && return MOI.OPTIMAL
    return MOI.get(s.inner, MOI.TerminationStatus())
end

function MOI.get(s::Optimizer, attr::MOI.PrimalStatus)
    s.use_cached && return MOI.FEASIBLE_POINT
    return MOI.get(s.inner, attr)
end

function MOI.get(s::Optimizer, attr::MOI.DualStatus)
    s.use_cached && return MOI.FEASIBLE_POINT
    return MOI.get(s.inner, attr)
end

function MOI.get(s::Optimizer, ::MOI.ResultCount)
    s.use_cached && return 1
    return MOI.get(s.inner, MOI.ResultCount())
end

function MOI.get(s::Optimizer, attr::MOI.ObjectiveValue)
    s.use_cached && return s.cached_obj_val
    return MOI.get(s.inner, attr)
end

function MOI.get(s::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    if s.use_cached
        slot = vi.value
        pos = slot <= length(s.cached_primal_pos) ? s.cached_primal_pos[slot] : 0
        return pos == 0 ? 0.0 : s.cached_primal[pos]
    end
    return MOI.get(s.inner, attr, vi)
end

function MOI.get(s::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex)
    if s.use_cached
        slot = ci.value
        pos = slot <= length(s.cached_dual_pos) ? s.cached_dual_pos[slot] : 0
        return pos == 0 ? 0.0 : s.cached_dual[pos]
    end
    return MOI.get(s.inner, attr, ci)
end

function MOI.optimize!(s::Optimizer)
    s.use_cached = false
    MOI.optimize!(s.inner)
end

DiffOpt.forward_differentiate!(s::Optimizer) = DiffOpt.forward_differentiate!(s.inner)
DiffOpt.reverse_differentiate!(s::Optimizer) = DiffOpt.reverse_differentiate!(s.inner)
DiffOpt.empty_input_sensitivities!(s::Optimizer) = DiffOpt.empty_input_sensitivities!(s.inner)
_get_sensitivity_cache(opt::Optimizer) = _get_sensitivity_cache(opt.inner)
_infer_optimizer_constructor(opt::Optimizer) = opt.optimizer_constructor
_highs_optimizer(opt::Optimizer) = _highs_optimizer(opt.inner)
_native_basis_backend(opt::Optimizer) = _native_basis_backend(opt.inner)
_is_diffopt_backend(::Optimizer) = true

function _objective_value(lp::LPData, x::AbstractVector, p::AbstractVector)
    total = lp.obj_c0
    @inbounds for i in eachindex(x)
        xi = x[i]
        total += lp.obj_c[i] * xi
        for j in eachindex(p)
            total += lp.obj_H[i, j] * p[j] * xi
        end
    end
    return lp.obj_scale * total
end
