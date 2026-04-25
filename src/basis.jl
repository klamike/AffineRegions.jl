function _build_certificate(lp::LPData, model::JuMP.Model; policy=default_policy(), tol=policy.basis_tol, exact_reduction=policy.exact_reduction)
    p = _get_param_vector(lp.parameters)
    basis_cols, z0, Zp, y0, Yp = _solve_basis_data(lp, model, p; tol)
    basic_set = Set(basis_cols)
    nonbasic_cols = Int[j for j in 1:size(lp.A, 2) if !(j in basic_set)]

    Bcols = lp.x_map[:, basis_cols]
    P_native = Matrix(Bcols * Zp)
    p0_native = Vector{Float64}(lp.x_offset .+ Bcols * z0)
    D_native, d0_native = _native_dual_law(lp, model, y0, Yp; tol)

    C_basic, c_basic = _select_region_rows(Zp, z0; tol=policy.zero_tol)

    if isempty(nonbasic_cols)
        C_red = zeros(Float64, 0, size(lp.F, 2))
        c_red = Float64[]
    else
        cN0 = lp.c[nonbasic_cols]
        HN = lp.H[nonbasic_cols, :]
        AtY = zeros(Float64, length(nonbasic_cols), size(lp.H, 2))
        active_y_cols = _active_param_cols(Yp; tol=policy.zero_tol)
        if !isempty(active_y_cols)
            @views AtY[:, active_y_cols] .= SparseArrays.transpose(lp.A[:, nonbasic_cols]) * Yp[:, active_y_cols]
        end
        Aty0 = SparseArrays.transpose(lp.A[:, nonbasic_cols]) * y0
        C_red, c_red = _select_region_rows(HN .- AtY, cN0 .- Aty0; tol=policy.zero_tol)
    end

    C = vcat(C_basic, C_red)
    c = vcat(c_basic, c_red)
    C, c = _reduce_constraints(C, c; policy, tol=policy.reduction_tol, exact=exact_reduction, optimizer_constructor=lp.optimizer_constructor)
    var_basis, row_basis, basis_hint =
        _capture_model_basis(model, lp.variables, _row_constraints(model))
    return BasisCertificate(
        copy(basis_cols),
        copy(p),
        C,
        c,
        P_native,
        p0_native,
        D_native,
        d0_native,
        var_basis,
        row_basis,
        basis_hint,
    )
end

function _build_canonical_solver(lp::LPData)
    model = JuMP.direct_model(HiGHS.Optimizer())
    JuMP.set_silent(model)
    n = size(lp.A, 2)
    m = size(lp.A, 1)
    z = Vector{JuMP.VariableRef}(undef, n)
    for j in 1:n
        z[j] = JuMP.@variable(model, lower_bound = 0.0)
    end

    At = copy(SparseArrays.transpose(lp.A))
    eq_refs = Vector{JuMP.ConstraintRef}(undef, m)
    for i in 1:m
        expr = JuMP.AffExpr(-lp.b[i])
        for ptr in SparseArrays.nzrange(At, i)
            col = At.rowval[ptr]
            coef = At.nzval[ptr]
            JuMP.add_to_expression!(expr, coef, z[col])
        end
        eq_refs[i] = JuMP.@constraint(model, expr == 0.0)
    end

    obj = JuMP.AffExpr(0.0)
    for j in 1:n
        lp.c[j] != 0.0 && JuMP.add_to_expression!(obj, lp.c[j], z[j])
    end
    JuMP.@objective(model, Min, obj)
    return CanonicalSolveCache(
        model,
        z,
        eq_refs,
        zeros(Float64, m),
        zeros(Float64, m),
        zeros(Float64, m),
        zeros(HiGHS.HighsInt, n),
        zeros(HiGHS.HighsInt, m),
        zeros(Cint, m),
        zeros(Float64, m),
        zeros(Float64, m, size(lp.F, 2)),
        zeros(Float64, m),
        zeros(Float64, m, size(lp.H, 2)),
    )
end

function _solve_basis_data(lp::LPData, primal_model::JuMP.Model, p::Vector{Float64}; tol=1e-8)
    cache = get!(primal_model.ext, :_AffineRegions_jl_CanonicalSolveCache) do
        _build_canonical_solver(lp)
    end
    if lp.rhs_parametric
        rhs = lp.b .+ lp.F * p
        cache.rhs .= rhs
        for i in eachindex(cache.eq_refs)
            JuMP.set_normalized_rhs(cache.eq_refs[i], rhs[i])
        end
    end
    if lp.obj_parametric
        c = lp.c .+ lp.H * p
        JuMP.set_objective_coefficient(cache.model, cache.z, c)
    end
    _apply_cached_basis!(cache)
    JuMP.optimize!(cache.model)
    JuMP.is_solved_and_feasible(cache.model) || throw(NativeBasisError("Internal standard-form basis solve failed."))
    _capture_basis!(cache)

    basic_cols = Int[]
    for bv in cache.basic_vars
        bv < 0 && continue
        push!(basic_cols, Int(bv) + 1)
    end
    if length(basic_cols) != size(lp.A, 1)
        basic_cols = _basis_columns_from_status(cache, tol)
    end
    fallback = length(basic_cols) != size(lp.A, 1)
    fallback && (basic_cols = _independent_basis_columns(lp.A, basic_cols; tol))

    z0 = cache.z0
    Zp = cache.Zp
    y0 = cache.y0
    Yp = cache.Yp

    if !fallback && length(basic_cols) == length(cache.basic_vars) &&
       all(cache.basic_vars[i] >= 0 && Int(cache.basic_vars[i]) + 1 == basic_cols[i] for i in eachindex(basic_cols))
        _basis_solve!(z0, cache, lp.b)
        fill!(Zp, 0.0)
        rhs_cols = lp.rhs_parametric ? _active_param_cols(lp.F; tol) : Int[]
        if !isempty(rhs_cols)
            for j in rhs_cols
                _basis_solve!(@view(Zp[:, j]), cache, @view(lp.F[:, j]))
            end
        end

        cB0 = lp.c[basic_cols]
        _basis_transpose_solve!(y0, cache, cB0)
        fill!(Yp, 0.0)
        obj_cols = lp.obj_parametric ? _active_param_cols(@view(lp.H[basic_cols, :]); tol) : Int[]
        if !isempty(obj_cols)
            for j in obj_cols
                _basis_transpose_solve!(@view(Yp[:, j]), cache, @view(lp.H[basic_cols, j]))
            end
        end
    else
        _solve_basis_data_fallback!(z0, Zp, y0, Yp, lp, basic_cols)
    end
    return basic_cols, copy(z0), copy(Zp), copy(y0), copy(Yp)
end

function _basis_solve!(dest::AbstractVector, cache::CanonicalSolveCache, rhs::AbstractVector)
    cache.basis_rhs .= rhs
    fill!(dest, 0.0)
    ret = HiGHS.Highs_getBasisSolve(cache_highs(cache).inner, cache.basis_rhs, dest, C_NULL, C_NULL)
    ret != 0 && throw(NativeBasisError("HiGHS basis solve failed with return code $ret"))
    return dest
end

function _basis_transpose_solve!(dest::AbstractVector, cache::CanonicalSolveCache, rhs::AbstractVector)
    cache.basis_rhs .= rhs
    fill!(dest, 0.0)
    ret = HiGHS.Highs_getBasisTransposeSolve(cache_highs(cache).inner, cache.basis_rhs, dest, C_NULL, C_NULL)
    ret != 0 && throw(NativeBasisError("HiGHS basis transpose solve failed with return code $ret"))
    return dest
end

cache_highs(cache::CanonicalSolveCache) = JuMP.backend(cache.model)::HiGHS.Optimizer

function _capture_basis!(cache::CanonicalSolveCache)
    highs = cache_highs(cache)
    ret = HiGHS.Highs_getBasis(highs.inner, cache.col_status, cache.row_status)
    ret != 0 && throw(NativeBasisError("HiGHS getBasis failed with return code $ret"))
    ret = HiGHS.Highs_getBasicVariables(highs.inner, cache.basic_vars)
    ret != 0 && throw(NativeBasisError("HiGHS getBasicVariables failed with return code $ret"))
    return cache
end

function _apply_cached_basis!(cache::CanonicalSolveCache)
    if isempty(cache.col_status) || isempty(cache.row_status)
        return
    end
    if any(status -> status != 0, cache.col_status) || any(status -> status != 0, cache.row_status)
        ret = HiGHS.Highs_setBasis(cache_highs(cache).inner, cache.col_status, cache.row_status)
        ret != 0 && throw(NativeBasisError("HiGHS setBasis failed with return code $ret"))
    end
    return
end

function _basis_columns_from_status(cache::CanonicalSolveCache, tol)
    basic_cols = Int[]
    for j in eachindex(cache.z)
        status = MOI.get(cache.model, MOI.VariableBasisStatus(), cache.z[j])
        if status == MOI.BASIC || status == MOI.SUPER_BASIC
            push!(basic_cols, j)
        end
    end
    if length(basic_cols) != length(cache.eq_refs)
        for j in eachindex(cache.z)
            length(basic_cols) == length(cache.eq_refs) && break
            j in basic_cols && continue
            rc = JuMP.reduced_cost(cache.z[j])
            abs(rc) <= tol && push!(basic_cols, j)
        end
    end
    return basic_cols
end

function _solve_basis_data_fallback!(z0, Zp, y0, Yp, lp::LPData, basis_cols::Vector{Int})
    B = lp.A[:, basis_cols]
    fac = LinearAlgebra.lu(B)
    z0 .= fac \ lp.b
    fill!(Zp, 0.0)
    rhs_cols = lp.rhs_parametric ? _active_param_cols(lp.F) : Int[]
    for j in rhs_cols
        @view(Zp[:, j]) .= fac \ @view(lp.F[:, j])
    end
    cB0 = lp.c[basis_cols]
    facT = LinearAlgebra.lu(SparseArrays.transpose(B))
    y0 .= facT \ cB0
    fill!(Yp, 0.0)
    obj_cols = lp.obj_parametric ? _active_param_cols(@view(lp.H[basis_cols, :])) : Int[]
    for j in obj_cols
        @view(Yp[:, j]) .= facT \ @view(lp.H[basic_cols, j])
    end
    return nothing
end

function _active_param_cols(M::AbstractMatrix; tol=1e-10)
    cols = Int[]
    for j in axes(M, 2)
        active = false
        for i in axes(M, 1)
            if abs(M[i, j]) > tol
                active = true
                break
            end
        end
        active && push!(cols, j)
    end
    return cols
end

function _select_region_rows(C::AbstractMatrix, c::AbstractVector; tol=1e-10)
    keep = Int[]
    for i in eachindex(c)
        active = false
        for j in axes(C, 2)
            if abs(C[i, j]) > tol
                active = true
                break
            end
        end
        (active || c[i] < -tol) && push!(keep, i)
    end
    isempty(keep) && return zeros(Float64, 0, size(C, 2)), Float64[]
    return Matrix(C[keep, :]), Vector{Float64}(c[keep])
end

function _capture_model_basis(model::JuMP.Model, variables::Vector{JuMP.VariableRef}, row_constraints::Vector{JuMP.ConstraintRef})
    backend = JuMP.backend(model)
    var_basis = Vector{MOI.BasisStatusCode}(undef, length(variables))
    for (i, vr) in enumerate(variables)
        var_basis[i] = MOI.get(backend, MOI.VariableBasisStatus(), JuMP.index(vr))
    end
    row_basis = Vector{MOI.BasisStatusCode}(undef, length(row_constraints))
    for (i, cr) in enumerate(row_constraints)
        row_basis[i] = MOI.get(backend, MOI.ConstraintBasisStatus(), JuMP.index(cr))
    end
    native = _native_basis_backend(backend)
    native === nothing && return var_basis, row_basis, NoBasisHint()
    return var_basis, row_basis, _capture_native_basis_hint(native, backend, variables, row_constraints)
end

function _native_dual_law(lp::LPData, model::JuMP.Model, y0::Vector{Float64}, Yp::Matrix{Float64}; tol=1e-9)
    n_dual = length(lp.duals)
    k = size(lp.obj_H, 2)
    D = zeros(Float64, n_dual, k)
    d0 = zeros(Float64, n_dual)

    n_rows = length(lp.row_signs)
    y0_rows = @view y0[1:n_rows]
    Yp_rows = @view Yp[1:n_rows, :]
    π0 = lp.row_signs .* y0_rows
    Π = Matrix(Yp_rows)
    for i in eachindex(lp.row_signs)
        @views Π[i, :] .*= lp.row_signs[i]
    end

    rc0 = lp.obj_c .- SparseArrays.transpose(lp.row_matrix) * π0
    Rc = lp.obj_H .- SparseArrays.transpose(lp.row_matrix) * Π
    statuses = _variable_basis_statuses(model, lp.variables)

    for i in 1:n_dual
        kind = lp.dual_kinds[i]
        src = lp.dual_sources[i]
        if kind == 0x01
            d0[i] = π0[src]
            @views D[i, :] .= Π[src, :]
        elseif kind == 0x02
            if statuses[src] == MOI.NONBASIC_AT_LOWER || statuses[src] == MOI.NONBASIC
                d0[i] = rc0[src]
                @views D[i, :] .= Rc[src, :]
            end
        elseif kind == 0x03
            if statuses[src] == MOI.NONBASIC_AT_UPPER
                d0[i] = rc0[src]
                @views D[i, :] .= Rc[src, :]
            end
        end
    end

    _snap_zeros!(D; tol)
    _snap_zeros!(d0; tol)
    return D, d0
end

function _variable_basis_statuses(model::JuMP.Model, variables::Vector{JuMP.VariableRef})
    backend = JuMP.backend(model)
    statuses = Vector{MOI.BasisStatusCode}(undef, length(variables))
    for (i, vr) in enumerate(variables)
        statuses[i] = MOI.get(backend, MOI.VariableBasisStatus(), JuMP.index(vr))
    end
    return statuses
end
