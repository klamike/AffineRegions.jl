_get_sensitivity_cache(model::JuMP.Model) = _get_sensitivity_cache(JuMP.backend(model))
_get_sensitivity_cache(opt::DiffOpt._SensitivityCache.Optimizer) = opt
_get_sensitivity_cache(opt::DiffOpt.Optimizer) = _get_sensitivity_cache(opt.optimizer)
_get_sensitivity_cache(opt::MOI.Utilities.CachingOptimizer) = _get_sensitivity_cache(opt.optimizer)
_get_sensitivity_cache(opt::MOI.Bridges.LazyBridgeOptimizer) = _get_sensitivity_cache(opt.model)
_get_sensitivity_cache(opt::POI.Optimizer) = _get_sensitivity_cache(opt.optimizer)
_get_sensitivity_cache(::Any) = nothing

_infer_optimizer_constructor(model::JuMP.Model) = _infer_optimizer_constructor(JuMP.backend(model))
_infer_optimizer_constructor(opt::DiffOpt._SensitivityCache.Optimizer) = typeof(opt.optimizer)
_infer_optimizer_constructor(opt::DiffOpt.Optimizer) = _infer_optimizer_constructor(opt.optimizer)
_infer_optimizer_constructor(opt::MOI.Utilities.CachingOptimizer) = _infer_optimizer_constructor(opt.optimizer)
_infer_optimizer_constructor(opt::MOI.Bridges.LazyBridgeOptimizer) = _infer_optimizer_constructor(opt.model)
_infer_optimizer_constructor(opt::POI.Optimizer) = _infer_optimizer_constructor(opt.optimizer)
_infer_optimizer_constructor(opt) = typeof(opt)

_highs_optimizer(model::JuMP.Model) = _highs_optimizer(JuMP.backend(model))
_highs_optimizer(opt::HiGHS.Optimizer) = opt
_highs_optimizer(opt::DiffOpt._SensitivityCache.Optimizer) = _highs_optimizer(opt.optimizer)
_highs_optimizer(opt::DiffOpt.Optimizer) = _highs_optimizer(opt.optimizer)
_highs_optimizer(opt::MOI.Utilities.CachingOptimizer) = _highs_optimizer(opt.optimizer)
_highs_optimizer(opt::MOI.Bridges.LazyBridgeOptimizer) = _highs_optimizer(opt.model)
_highs_optimizer(opt::POI.Optimizer) = _highs_optimizer(opt.optimizer)
_highs_optimizer(::Any) = nothing

abstract type NativeBasisBackend end
abstract type NativeBasisHint end

struct NoBasisHint <: NativeBasisHint end
struct HiGHSBasisBackend <: NativeBasisBackend end

struct HiGHSBasisHint <: NativeBasisHint
    col_status::Vector{HiGHS.HighsInt}
    row_status::Vector{HiGHS.HighsInt}
end

struct MOIBasisHint <: NativeBasisHint
    var_status::Vector{MOI.BasisStatusCode}
    row_status::Vector{MOI.BasisStatusCode}
end

_native_basis_backend(model::JuMP.Model) = _native_basis_backend(JuMP.backend(model))
_native_basis_backend(opt::HiGHS.Optimizer) = HiGHSBasisBackend()
_native_basis_backend(opt::DiffOpt._SensitivityCache.Optimizer) = _native_basis_backend(opt.optimizer)
_native_basis_backend(opt::DiffOpt.Optimizer) = _native_basis_backend(opt.optimizer)
_native_basis_backend(opt::MOI.Utilities.CachingOptimizer) = _native_basis_backend(opt.optimizer)
_native_basis_backend(opt::MOI.Bridges.LazyBridgeOptimizer) = _native_basis_backend(opt.model)
_native_basis_backend(opt::POI.Optimizer) = _native_basis_backend(opt.optimizer)
_native_basis_backend(::Any) = nothing

_native_basis_name(::HiGHSBasisBackend) = "HiGHS"

function _capture_native_basis_data(::HiGHSBasisBackend, opt)
    highs = _highs_optimizer(opt)
    n_col = Int(HiGHS.Highs_getNumCol(highs.inner))
    n_row = Int(HiGHS.Highs_getNumRow(highs.inner))
    col_status = zeros(HiGHS.HighsInt, n_col)
    row_status = zeros(HiGHS.HighsInt, n_row)
    ret = HiGHS.Highs_getBasis(highs.inner, col_status, row_status)
    ret == 0 || throw(NativeBasisError("HiGHS getBasis failed with return code $ret"))
    return col_status, row_status
end

function _capture_native_basis_hint(::HiGHSBasisBackend, opt, _variables, _row_constraints)
    col_status, row_status = _capture_native_basis_data(HiGHSBasisBackend(), opt)
    return HiGHSBasisHint(col_status, row_status)
end

function _apply_native_basis_data!(::HiGHSBasisBackend, opt, col_status::Vector{HiGHS.HighsInt}, row_status::Vector{HiGHS.HighsInt})
    highs = _highs_optimizer(opt)
    length(col_status) == Int(HiGHS.Highs_getNumCol(highs.inner)) || return false
    length(row_status) == Int(HiGHS.Highs_getNumRow(highs.inner)) || return false
    ret = HiGHS.Highs_setBasis(highs.inner, col_status, row_status)
    ret == 0 || return false
    return true
end

function _apply_native_basis_hint!(::HiGHSBasisBackend, opt, _variables, _row_constraints, hint::HiGHSBasisHint)
    return _apply_native_basis_data!(HiGHSBasisBackend(), opt, hint.col_status, hint.row_status)
end

_capture_native_basis_hint(::NativeBasisBackend, _opt, variables, row_constraints) = MOIBasisHint(MOI.BasisStatusCode[], MOI.BasisStatusCode[])
_apply_native_basis_hint!(::NativeBasisBackend, _opt, _variables, _row_constraints, ::NativeBasisHint) = false

_constructor_type(constructor::DataType) = constructor
_constructor_type(constructor) = typeof(constructor())

is_parameter_constraint(
    ::JuMP.ConstraintRef{M,MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{T}}},
) where {M,T} = true
is_parameter_constraint(::JuMP.ConstraintRef{M,MOI.ConstraintIndex{F,S}}) where {M,F,S} = false

all_constraints(model) = JuMP.all_constraints(model, include_variable_in_set_constraints = true)

function _classify_model_entities(model::JuMP.Model)
    xp = JuMP.all_variables(model)
    cons = all_constraints(model)
    variables = filter(!JuMP.is_parameter, xp)
    parameters = filter(JuMP.is_parameter, xp)
    duals = filter(!is_parameter_constraint, cons)
    return variables, parameters, duals
end

function _collect_row_constraints(model::JuMP.Model)
    refs = JuMP.ConstraintRef[]
    for c in all_constraints(model)
        is_parameter_constraint(c) && continue
        idx = JuMP.index(c)
        idx isa MOI.ConstraintIndex{MOI.VariableIndex} && continue
        co = JuMP.constraint_object(c)
        if co.func isa Union{JuMP.GenericAffExpr,JuMP.VariableRef,Number} &&
           co.set isa Union{MOI.EqualTo,MOI.LessThan,MOI.GreaterThan}
            push!(refs, c)
        else
            throw(UnsupportedModelError("Unsupported constraint in AffineRegions: $(typeof(co.func)) in $(typeof(co.set))."))
        end
    end
    return refs
end

function _extract_objective_data(model, variables, parameters, var_idx, param_idx)
    obj = JuMP.objective_function(model)
    c = zeros(Float64, length(variables))
    H = zeros(Float64, length(variables), length(parameters))
    c0 = 0.0
    _accumulate_objective!(c, H, Ref(c0), obj, var_idx, param_idx)
    return c, H, c0
end

function _accumulate_objective!(c, H, c0_ref::Base.RefValue{Float64}, obj::Number, var_idx, param_idx)
    c0_ref[] += Float64(obj)
    return
end

function _accumulate_objective!(c, H, c0_ref::Base.RefValue{Float64}, obj::JuMP.VariableRef, var_idx, param_idx)
    if JuMP.is_parameter(obj)
        return
    end
    c[var_idx[obj]] += 1.0
    return
end

function _accumulate_objective!(c, H, c0_ref::Base.RefValue{Float64}, obj::JuMP.GenericAffExpr, var_idx, param_idx)
    c0_ref[] += obj.constant
    for (var, coef) in obj.terms
        if var isa JuMP.VariableRef
            if JuMP.is_parameter(var)
                continue
            end
            c[var_idx[var]] += coef
        else
            c0_ref[] += coef * Float64(var)
        end
    end
    return
end

function _accumulate_objective!(c, H, c0_ref::Base.RefValue{Float64}, obj::JuMP.GenericQuadExpr, var_idx, param_idx)
    _accumulate_objective!(c, H, c0_ref, obj.aff, var_idx, param_idx)
    for (coef, v1, v2) in JuMP.quad_terms(obj)
        if JuMP.is_parameter(v1) && !JuMP.is_parameter(v2)
            H[var_idx[v2], param_idx[v1]] += coef
        elseif JuMP.is_parameter(v2) && !JuMP.is_parameter(v1)
            H[var_idx[v1], param_idx[v2]] += coef
        else
            throw(UnsupportedModelError("Unsupported objective term in AffineRegions: quadratic terms must be parameter-variable products."))
        end
    end
    return
end

function _extract_constraint_parts(func::Number, var_idx, param_idx)
    return zeros(Float64, length(var_idx)), zeros(Float64, length(param_idx)), Float64(func)
end

function _extract_constraint_parts(func::JuMP.VariableRef, var_idx, param_idx)
    a = zeros(Float64, length(var_idx))
    g = zeros(Float64, length(param_idx))
    if JuMP.is_parameter(func)
        g[param_idx[func]] = 1.0
    else
        a[var_idx[func]] = 1.0
    end
    return a, g, 0.0
end

function _extract_constraint_parts(func::JuMP.GenericAffExpr, var_idx, param_idx)
    a = zeros(Float64, length(var_idx))
    g = zeros(Float64, length(param_idx))
    const_term = func.constant
    for (var, coef) in func.terms
        if var isa JuMP.VariableRef
            if JuMP.is_parameter(var)
                g[param_idx[var]] += coef
            else
                a[var_idx[var]] += coef
            end
        else
            const_term += coef * Float64(var)
        end
    end
    return a, g, const_term
end

function _normalized_row(func, set::MOI.LessThan, var_idx, param_idx)
    a, g, const_term = _extract_constraint_parts(func, var_idx, param_idx)
    rhs0 = set.upper - const_term
    return a, -g, rhs0, false
end

function _normalized_row(func, set::MOI.GreaterThan, var_idx, param_idx)
    a, g, const_term = _extract_constraint_parts(func, var_idx, param_idx)
    rhs0 = -set.lower + const_term
    return -a, g, rhs0, false
end

function _normalized_row(func, set::MOI.EqualTo, var_idx, param_idx)
    a, g, const_term = _extract_constraint_parts(func, var_idx, param_idx)
    rhs0 = set.value - const_term
    return a, -g, rhs0, true
end

function compute_laws(model; tol=1e-10)
    variables, parameters, duals = get!(model.ext, :_AffineRegions_jl_VarClass) do
        _classify_model_entities(model)
    end

    var_idx, param_idx = get!(model.ext, :_AffineRegions_jl_IdxMaps) do
        (
            Dict{JuMP.VariableRef,Int}(v => i for (i, v) in enumerate(variables)),
            Dict{JuMP.VariableRef,Int}(p => i for (i, p) in enumerate(parameters)),
        )
    end

    vi_to_row, ci_to_row = get!(model.ext, :_AffineRegions_jl_MOIIdxMaps) do
        (
            Dict{MOI.VariableIndex,Int}(JuMP.index(vr) => i for (i, vr) in enumerate(variables)),
            Dict{MOI.ConstraintIndex,Int}(JuMP.index(cr) => i for (i, cr) in enumerate(duals)),
        )
    end

    n, n_d, k = length(variables), length(duals), length(parameters)
    P, p0, D, d0 = get!(model.ext, :_AffineRegions_jl_LawBuffers) do
        (zeros(Float64, n, k), zeros(Float64, n), zeros(Float64, n_d, k), zeros(Float64, n_d))
    end
    fill!(P, 0.0)
    fill!(p0, 0.0)
    fill!(D, 0.0)
    fill!(d0, 0.0)

    sc = _get_sensitivity_cache(model)
    for (j, p_j) in enumerate(parameters)
        DiffOpt.empty_input_sensitivities!(model)
        for p_i in parameters
            DiffOpt.set_forward_parameter(model, p_i, p_i === p_j ? 1.0 : 0.0)
        end
        DiffOpt.forward_differentiate!(model)

        if sc !== nothing && sc.forw_dx !== nothing
            for (vi, val) in sc.forw_dx
                row = get(vi_to_row, vi, 0)
                row > 0 && (P[row, j] = val)
            end
        else
            for (i, x_i) in enumerate(variables)
                P[i, j] = MOI.get(model, DiffOpt.ForwardVariablePrimal(), x_i)
            end
        end

        if sc !== nothing && sc.forw_dy !== nothing
            for (ci, val) in sc.forw_dy
                row = get(ci_to_row, ci, 0)
                row > 0 && (D[row, j] = val)
            end
        else
            for (i, λ_i) in enumerate(duals)
                D[i, j] = MOI.get(model, DiffOpt.ForwardConstraintDual(), λ_i)
            end
        end
    end

    param_vals = _get_param_vector(parameters)
    for (i, x_i) in enumerate(variables)
        p0[i] = JuMP.value(x_i)
    end
    for (i, λ_i) in enumerate(duals)
        d0[i] = JuMP.dual(λ_i)
    end
    for j in 1:k
        pv = param_vals[j]
        for i in 1:n
            p0[i] -= P[i, j] * pv
        end
        for i in 1:n_d
            d0[i] -= D[i, j] * pv
        end
    end

    _snap_zeros!(P; tol)
    _snap_zeros!(p0; tol)
    _snap_zeros!(D; tol)
    _snap_zeros!(d0; tol)
    return P, p0, D, d0
end

function _laws_to_dicts(P, p0, D, d0, model; tol=1e-10)
    variables, parameters, duals = get!(model.ext, :_AffineRegions_jl_VarClass) do
        _classify_model_entities(model)
    end

    primal_laws = Dict{JuMP.VariableRef,JuMP.AffExpr}()
    for (i, x_i) in enumerate(variables)
        expr = JuMP.AffExpr(p0[i])
        for (j, p_j) in enumerate(parameters)
            abs(P[i, j]) > tol && JuMP.add_to_expression!(expr, P[i, j], p_j)
        end
        primal_laws[x_i] = expr
    end

    dual_laws = Dict{JuMP.ConstraintRef,JuMP.AffExpr}()
    for (i, λ_i) in enumerate(duals)
        expr = JuMP.AffExpr(d0[i])
        for (j, p_j) in enumerate(parameters)
            abs(D[i, j]) > tol && JuMP.add_to_expression!(expr, D[i, j], p_j)
        end
        dual_laws[λ_i] = expr
    end
    return primal_laws, dual_laws
end

function _matrix_to_constraints(C, c, parameters; tol=1e-10)
    constraints = JuMP.ScalarConstraint[]
    for i in axes(C, 1)
        expr = JuMP.AffExpr(c[i])
        for j in axes(C, 2)
            abs(C[i, j]) > tol && JuMP.add_to_expression!(expr, C[i, j], parameters[j])
        end
        push!(constraints, JuMP.ScalarConstraint(expr, MOI.GreaterThan(0.0)))
    end
    return constraints
end

function _canonical_row_key(a::Vector{Float64}, p_rhs::Vector{Float64}, rhs0::Float64, is_eq::Bool; tol=1e-9)
    vec = vcat(a, p_rhs, [rhs0])
    scale = maximum(abs, vec)
    scale = max(scale, 1.0)
    vec ./= scale
    pivot = findfirst(v -> abs(v) > tol, vec)
    if pivot !== nothing && vec[pivot] < 0
        vec .*= -1.0
    end
    return string(is_eq, "|", join([string(round(Int, v / tol)) for v in vec], ","))
end

function _independent_basis_columns(A::SparseArrays.SparseMatrixCSC{Float64,Int}, candidates::Vector{Int}; tol=1e-8)
    m = size(A, 1)
    chosen = Int[]
    M = zeros(Float64, m, 0)
    for j in candidates
        col = Matrix(A[:, [j]])
        trial = hcat(M, col)
        if LinearAlgebra.rank(trial; atol=tol) > size(M, 2)
            push!(chosen, j)
            M = trial
        end
        length(chosen) == m && return chosen
    end
    for j in 1:size(A, 2)
        j in chosen && continue
        col = Matrix(A[:, [j]])
        trial = hcat(M, col)
        if LinearAlgebra.rank(trial; atol=tol) > size(M, 2)
            push!(chosen, j)
            M = trial
        end
        length(chosen) == m && return chosen
    end
    throw(NativeBasisError("Unable to construct an independent basis column set."))
end

function _reduce_constraints(C::Matrix{Float64}, c::Vector{Float64}; policy=default_policy(), tol=policy.reduction_tol, exact=policy.exact_reduction, optimizer_constructor=nothing)
    size(C, 1) == length(c) || throw(InvalidModelStateError("Constraint matrix/vector size mismatch during region reduction."))
    if isempty(c)
        return C, c
    end

    keep_C = Vector{Vector{Float64}}()
    keep_c = Float64[]
    key_to_idx = Dict{String,Int}()
    k = size(C, 2)

    for i in axes(C, 1)
        row = collect(@view C[i, :])
        rhs = c[i]
        if all(abs(v) <= tol for v in row)
            rhs < -tol && (push!(keep_C, zeros(Float64, k)); push!(keep_c, rhs))
            continue
        end

        scale = maximum(abs, row)
        scale = max(scale, 1.0)
        row ./= scale
        rhs /= scale

        _snap_zeros!(row; tol)
        abs(rhs) < tol && (rhs = 0.0)

        key = join([string(round(Int, v / tol)) for v in row], ",")
        if haskey(key_to_idx, key)
            idx = key_to_idx[key]
            if rhs < keep_c[idx]
                keep_C[idx] = row
                keep_c[idx] = rhs
            end
        else
            push!(keep_C, row)
            push!(keep_c, rhs)
            key_to_idx[key] = length(keep_c)
        end
    end

    C2 = isempty(keep_C) ? zeros(Float64, 0, k) : reduce(vcat, permutedims.(keep_C))
    c2 = copy(keep_c)

    if exact &&
       k <= policy.exact_reduction_param_limit &&
       length(c2) <= policy.exact_reduction_row_limit &&
       optimizer_constructor !== nothing
        keep = trues(length(c2))
        for i in eachindex(c2)
            keep[i] || continue
            redundant = _row_is_redundant(C2, c2, i, optimizer_constructor; tol)
            redundant && (keep[i] = false)
        end
        C2 = C2[keep, :]
        c2 = c2[keep]
    end

    return C2, c2
end

function _row_is_redundant(C::Matrix{Float64}, c::Vector{Float64}, row_idx::Int, optimizer_constructor; tol=1e-9)
    size(C, 2) == 0 && return c[row_idx] >= -tol
    model = JuMP.Model(optimizer_constructor)
    JuMP.set_silent(model)
    p = [JuMP.@variable(model) for _ in 1:size(C, 2)]
    for i in eachindex(c)
        i == row_idx && continue
        expr = JuMP.AffExpr(c[i])
        for j in axes(C, 2)
            abs(C[i, j]) > tol && JuMP.add_to_expression!(expr, C[i, j], p[j])
        end
        JuMP.@constraint(model, expr >= 0.0)
    end
    obj = JuMP.AffExpr(c[row_idx])
    for j in axes(C, 2)
        abs(C[row_idx, j]) > tol && JuMP.add_to_expression!(obj, C[row_idx, j], p[j])
    end
    JuMP.@objective(model, Min, obj)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    if status == MOI.OPTIMAL
        return JuMP.objective_value(model) >= -tol
    end
    return false
end

function _get_param_vector(parameters)
    return Float64[JuMP.parameter_value(p) for p in parameters]
end

function _get_param_vector!(buffer::Vector{Float64}, parameters)
    length(buffer) == length(parameters) || resize!(buffer, length(parameters))
    for i in eachindex(parameters)
        buffer[i] = JuMP.parameter_value(parameters[i])
    end
    return buffer
end

function _set_param_vector!(parameters, p::AbstractVector)
    length(parameters) == length(p) || throw(DimensionMismatch("Parameter vector has length $(length(p)); expected $(length(parameters))."))
    for i in eachindex(parameters)
        JuMP.set_parameter_value(parameters[i], p[i])
    end
    return parameters
end

function _snap_zeros!(A::AbstractArray; tol=1e-10)
    for i in eachindex(A)
        abs(A[i]) < tol && (A[i] = 0.0)
    end
    return A
end

_is_diffopt_backend(::DiffOpt.Optimizer) = true
_is_diffopt_backend(::Any) = false
