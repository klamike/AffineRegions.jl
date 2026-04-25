check_model(model) = begin
    !_is_diffopt_backend(JuMP.backend(model)) && (
        throw(UnsupportedModelError("Backend must be a DiffOpt.Optimizer-compatible optimizer. Got: $(JuMP.backend(model))"))
    )
    !JuMP.is_solved_and_feasible(model) && (
        throw(InvalidModelStateError("Model must be solved and feasible before building a certificate. Got termination status $(JuMP.termination_status(model))."))
    )
    _native_basis_backend(JuMP.backend(model)) === nothing && (
        throw(UnsupportedModelError("AffineRegions currently supports solvers with native basis-status access only. Implemented backends: HiGHS and optional extensions."))
    )
end

function extract_lp(model::JuMP.Model)
    return get!(model.ext, :_AffineRegions_jl_LPData) do
        _extract_lp_data(model)
    end
end

_row_constraints(model::JuMP.Model) = get!(model.ext, :_AffineRegions_jl_RowConstraints) do
    _collect_row_constraints(model)
end

function _extract_lp_data(model::JuMP.Model; tol=1e-10)
    sense = JuMP.objective_sense(model)
    sense in (MOI.MIN_SENSE, MOI.MAX_SENSE) ||
        throw(UnsupportedModelError("AffineRegions supports linear minimization and maximization LPs only."))
    obj_scale = sense == MOI.MAX_SENSE ? -1.0 : 1.0

    variables, parameters, duals = _classify_model_entities(model)
    n = length(variables)
    k = length(parameters)
    var_idx = Dict{JuMP.VariableRef,Int}(v => i for (i, v) in enumerate(variables))
    param_idx = Dict{JuMP.VariableRef,Int}(p => i for (i, p) in enumerate(parameters))

    c_orig, H_orig, c0 = _extract_objective_data(model, variables, parameters, var_idx, param_idx)
    c_orig .*= obj_scale
    H_orig .*= obj_scale
    c0 *= obj_scale

    var_reps = Vector{VarRep}(undef, n)
    x_offset = zeros(Float64, n)
    xmap_I = Int[]
    xmap_J = Int[]
    xmap_V = Float64[]
    ub_rows = Tuple{Int,Int,Float64}[]
    next_col = 0

    for (i, v) in enumerate(variables)
        has_lb = JuMP.has_lower_bound(v)
        has_ub = JuMP.has_upper_bound(v)
        lb = has_lb ? JuMP.lower_bound(v) : 0.0
        ub = has_ub ? JuMP.upper_bound(v) : 0.0

        if has_lb && has_ub && abs(lb - ub) <= tol
            x_offset[i] = lb
            var_reps[i] = VarRep(lb, Int[], Float64[])
            continue
        elseif has_lb && has_ub
            next_col += 1
            w = next_col
            next_col += 1
            t = next_col
            x_offset[i] = lb
            push!(xmap_I, i); push!(xmap_J, w); push!(xmap_V, 1.0)
            var_reps[i] = VarRep(lb, [w], [1.0])
            push!(ub_rows, (w, t, ub - lb))
        elseif has_lb
            next_col += 1
            w = next_col
            x_offset[i] = lb
            push!(xmap_I, i); push!(xmap_J, w); push!(xmap_V, 1.0)
            var_reps[i] = VarRep(lb, [w], [1.0])
        elseif has_ub
            next_col += 1
            w = next_col
            x_offset[i] = ub
            push!(xmap_I, i); push!(xmap_J, w); push!(xmap_V, -1.0)
            var_reps[i] = VarRep(ub, [w], [-1.0])
        else
            next_col += 1
            pos = next_col
            next_col += 1
            neg = next_col
            push!(xmap_I, i); push!(xmap_J, pos); push!(xmap_V, 1.0)
            push!(xmap_I, i); push!(xmap_J, neg); push!(xmap_V, -1.0)
            var_reps[i] = VarRep(0.0, [pos, neg], [1.0, -1.0])
        end
    end

    n_std = next_col
    c_std = zeros(Float64, n_std)
    H_std = zeros(Float64, n_std, k)
    for i in 1:n
        rep = var_reps[i]
        for t in eachindex(rep.cols)
            col = rep.cols[t]
            sign = rep.vals[t]
            c_std[col] += sign * c_orig[i]
            for q in 1:k
                H_std[col, q] += sign * H_orig[i, q]
            end
        end
    end

    row_I = Int[]
    row_J = Int[]
    row_V = Float64[]
    row_orig_I = Int[]
    row_orig_J = Int[]
    row_orig_V = Float64[]
    b = Float64[]
    row_rhs = Vector{Vector{Float64}}()
    row_count = 0
    row_signs = Float64[]
    row_key_to_idx = Dict{String,Int}()
    dual_pos = Dict{JuMP.ConstraintRef,Int}(cr => i for (i, cr) in enumerate(duals))
    dual_kinds = fill(UInt8(0), length(duals))
    dual_sources = zeros(Int, length(duals))

    for cref in _row_constraints(model)
        co = JuMP.constraint_object(cref)
        a_orig, p_rhs, rhs0, is_eq = _normalized_row(co.func, co.set, var_idx, param_idx)
        row_key = _canonical_row_key(a_orig, p_rhs, rhs0, is_eq; tol)
        row_sign = co.set isa MOI.GreaterThan ? -1.0 : 1.0
        if haskey(row_key_to_idx, row_key)
            dual_idx = get(dual_pos, cref, 0)
            if dual_idx > 0
                dual_kinds[dual_idx] = 0x01
                dual_sources[dual_idx] = row_key_to_idx[row_key]
            end
            continue
        end
        row_count += 1
        row_key_to_idx[row_key] = row_count
        push!(row_signs, row_sign)
        rhs0 -= sum(a_orig[i] * x_offset[i] for i in 1:n)
        push!(b, rhs0)
        push!(row_rhs, p_rhs)
        for i in 1:n
            ai = a_orig[i]
            abs(ai) <= tol && continue
            push!(row_orig_I, row_count)
            push!(row_orig_J, i)
            push!(row_orig_V, ai)
            rep = var_reps[i]
            for t in eachindex(rep.cols)
                push!(row_I, row_count)
                push!(row_J, rep.cols[t])
                push!(row_V, ai * rep.vals[t])
            end
        end
        if !is_eq
            next_col += 1
            slack_col = next_col
            push!(row_I, row_count)
            push!(row_J, slack_col)
            push!(row_V, 1.0)
        end
        dual_idx = get(dual_pos, cref, 0)
        if dual_idx > 0
            dual_kinds[dual_idx] = 0x01
            dual_sources[dual_idx] = row_count
        end
    end

    for (w, t, width) in ub_rows
        row_count += 1
        push!(b, width)
        push!(row_rhs, zeros(Float64, k))
        push!(row_I, row_count); push!(row_J, w); push!(row_V, 1.0)
        push!(row_I, row_count); push!(row_J, t); push!(row_V, 1.0)
    end

    F = zeros(Float64, row_count, k)
    for i in 1:row_count
        F[i, :] .= row_rhs[i]
    end
    rhs_parametric = any(abs(v) > tol for v in F)

    if next_col != n_std
        c_aug = zeros(Float64, next_col)
        c_aug[1:length(c_std)] .= c_std
        c_std = c_aug
        H_aug = zeros(Float64, next_col, k)
        H_aug[1:size(H_std, 1), :] .= H_std
        H_std = H_aug
        n_std = next_col
    end
    obj_parametric = any(abs(v) > tol for v in H_std)

    A = SparseArrays.sparse(row_I, row_J, row_V, row_count, n_std)
    row_matrix = SparseArrays.sparse(row_orig_I, row_orig_J, row_orig_V, length(row_signs), n)
    x_map = SparseArrays.sparse(xmap_I, xmap_J, xmap_V, n, n_std)
    for (i, v) in enumerate(variables)
        if JuMP.has_lower_bound(v)
            idx = get(dual_pos, JuMP.LowerBoundRef(v), 0)
            idx > 0 && (dual_kinds[idx] = 0x02; dual_sources[idx] = i)
        end
        if JuMP.has_upper_bound(v)
            idx = get(dual_pos, JuMP.UpperBoundRef(v), 0)
            idx > 0 && (dual_kinds[idx] = 0x03; dual_sources[idx] = i)
        end
    end
    return LPData(
        variables,
        parameters,
        duals,
        c_orig,
        H_orig,
        c0,
        obj_scale,
        row_matrix,
        row_signs,
        dual_kinds,
        dual_sources,
        A,
        F,
        rhs_parametric,
        b,
        c_std,
        H_std,
        obj_parametric,
        c0,
        x_offset,
        x_map,
        var_reps,
        _infer_optimizer_constructor(model),
    )
end
