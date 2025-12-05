module AffineRegions

import JuMP
import DiffOpt
import Dualization

const MOI = JuMP.MOI
const DualizationJuMPExt = Base.get_extension(Dualization, :DualizationJuMPExt)

export affine_region
function affine_region(model::JuMP.Model)
    check_model(model)

    primal_laws, dual_laws = compute_laws(model)
    constraints = compute_constraints(model, primal_laws, dual_laws)

    return (
        constraints,   # constraints defining the region
        primal_laws,   # primal law within region
        dual_laws,     # dual law within region
    )
end


function compute_laws(model; tol=1e-10)
    xp = JuMP.all_variables(model)
    all_cons = JuMP.all_constraints(model, include_variable_in_set_constraints = true)

    variables = filter(!JuMP.is_parameter, xp)
    parameters = filter(JuMP.is_parameter, xp)
    duals = filter(!is_parameter_constraint, all_cons)

    primal_laws = Dict{JuMP.VariableRef, JuMP.AffExpr}(
        xᵢ => JuMP.AffExpr() for xᵢ in variables
    )
    dual_laws = Dict{JuMP.ConstraintRef, JuMP.AffExpr}(
        λᵢ => JuMP.AffExpr() for λᵢ in duals
    )
    for pⱼ in parameters
        # set sensitivities
        DiffOpt.empty_input_sensitivities!(model)
        for pᵢ in parameters
            if pᵢ === pⱼ
                DiffOpt.set_forward_parameter(model, pᵢ, 1.0)
            else
                DiffOpt.set_forward_parameter(model, pᵢ, 0.0)
            end
        end

        # compute sensitivites
        DiffOpt.forward_differentiate!(model)

        # extract primal sensitivities
        for xᵢ in variables
            δ_xᵢpⱼ = MOI.get(model, DiffOpt.ForwardVariablePrimal(), xᵢ)
            abs(δ_xᵢpⱼ) < tol && continue
            JuMP.add_to_expression!(primal_laws[xᵢ], δ_xᵢpⱼ, pⱼ)
        end

        # extract dual sensitivities
        for λᵢ in duals
            δ_λᵢpⱼ = MOI.get(model, DiffOpt.ForwardConstraintDual(), λᵢ)
            abs(δ_λᵢpⱼ) < tol && continue
            JuMP.add_to_expression!(dual_laws[λᵢ], δ_λᵢpⱼ, pⱼ)
        end
    end

    # compute primal offset
    for xᵢ in variables
        law = primal_laws[xᵢ]
        x = JuMP.value(xᵢ)
        v = JuMP.value(law)
        JuMP.add_to_expression!(law, x - v)
    end

    # compute dual offset
    for λᵢ in duals
        law = dual_laws[λᵢ]
        λ = JuMP.dual(λᵢ)
        v = JuMP.value(law)
        JuMP.add_to_expression!(law, λ - v)
    end
    return primal_laws, dual_laws
end

function compute_constraints(model, primal_laws, dual_laws;
                             force_new_dual = true,  # TODO: detect when only parameters changed
                             add_dual_constraints = true,
                             strong_duality_set = nothing)  # e.g. MOI.EqualTo(0.0)
    constraints = JuMP.ScalarConstraint[]

    # substitute primal law into primal constraints
    primal_constraints = JuMP.all_constraints(model, include_variable_in_set_constraints = true)
    for constraint in primal_constraints
        is_parameter_constraint(constraint) && continue
        is_equality_constraint(constraint) && continue

        co = JuMP.constraint_object(constraint)
        func = JuMP.value(vr -> get_primal_law(vr, primal_laws), co.func)
        JuMP.drop_zeros!(func)
        should_skip(func, co.set) && continue
        push!(constraints, JuMP.ScalarConstraint(func, co.set))
    end

    if add_dual_constraints || !isnothing(strong_duality_set)

        # FIXME: dualization requires AUTOMATIC mode
        dual_model, buffer_map = if force_new_dual || !haskey(model.ext, :_AffineRegions_jl_DualModel)
            buffer_model = JuMP.Model()
            buffer_map = MOI.copy_to(JuMP.backend(buffer_model), model)
            dual_model = Dualization.dualize(buffer_model)
            model.ext[:_AffineRegions_jl_DualModel] = (dual_model, buffer_map)
        else
            model.ext[:_AffineRegions_jl_DualModel]
        end

        if add_dual_constraints
            # substitute dual law into dual constraints
            dual_constraints = JuMP.all_constraints(dual_model, include_variable_in_set_constraints = true)
            for constraint in dual_constraints
                is_parameter_constraint(constraint) && continue
                is_equality_constraint(constraint) && continue

                co = JuMP.constraint_object(constraint)
                func = JuMP.value(vr -> get_dual_law(vr, dual_model, dual_laws, primal_laws, buffer_map, model), co.func)
                JuMP.drop_zeros!(func)
                should_skip(func, co.set) && continue
                push!(constraints, JuMP.ScalarConstraint(func, co.set))
            end
        end

        if !isnothing(strong_duality_set)
            # substitute laws into (primal_obj = dual_obj)
            primal_obj = JuMP.objective_function(model)
            dual_obj = JuMP.objective_function(dual_model)

            primal_value = JuMP.value(vr -> get_primal_law(vr, primal_laws), primal_obj)
            dual_value = JuMP.value(vr -> get_dual_law(vr, dual_model, dual_laws, primal_laws, buffer_map, model), dual_obj)

            func = primal_value - dual_value
            is_aff_or_quad(func) && JuMP.drop_zeros!(func)
            !should_skip(func, co.set) && push!(constraints, JuMP.ScalarConstraint(func, strong_duality_set))
        end
    end

    return constraints
end


function get_primal_law(vr::JuMP.VariableRef, primal_laws)::JuMP.AffExpr
    if JuMP.is_parameter(vr)
        return vr + 0
    end

    if haskey(primal_laws, vr)
        return primal_laws[vr]
    end

    error("Could not find primal variable $vr.")
end

# FIXME: DualPrimalMap?
function get_dual_law(vr::JuMP.VariableRef, dual_model::JuMP.Model, dual_laws, primal_laws, buffer_map, primal_model)::JuMP.AffExpr
    pdm = DualizationJuMPExt._get_primal_dual_map(dual_model)

    if JuMP.is_parameter(vr)
        return vr + 0
    end

    for (k, v) in pdm.primal_constraint_data
        if JuMP.index(vr) in v.dual_variables
            cr = JuMP.constraint_ref_with_index(primal_model, buffer_map[k])
            return dual_laws[cr]
        end
    end

    for (k, v) in pdm.primal_var_in_quad_obj_to_dual_slack_var
        if JuMP.index(vr) === v
            return primal_laws[buffer_map[k]]
        end
    end

    error("Could not find dual variable $vr.")
end


is_parameter_constraint(
    ::JuMP.ConstraintRef{M, MOI.ConstraintIndex{MOI.VariableIndex, MOI.Parameter{T}}}
) where {M, T} = true

is_parameter_constraint(
    ::JuMP.ConstraintRef{M, MOI.ConstraintIndex{F, S}}
) where {M, F, S} = false

is_equality_constraint(
    ::JuMP.ConstraintRef{M, MOI.ConstraintIndex{F, MOI.EqualTo{T}}}
) where {M, F, T} = true

is_equality_constraint(
    ::JuMP.ConstraintRef{M, MOI.ConstraintIndex{F, S}}
) where {M, F, S} = false


function should_skip(func::JuMP.AffExpr, set)
    !isempty(func.terms) && return false
    return _should_skip(func, set)
end

function should_skip(func::JuMP.QuadExpr, set)
    !(isempty(func.aff.terms) && isempty(func.terms)) && return false
    return _should_skip(func, set)
end

should_skip(func, set) = false
_should_skip(func, set) = false
_should_skip(func, set::MOI.EqualTo) = (JuMP.value(func) == set.value)
_should_skip(func, set::MOI.GreaterThan) = (JuMP.value(func) >= set.lower)
_should_skip(func, set::MOI.LessThan) = (JuMP.value(func) <= set.upper)

is_aff_or_quad(func::JuMP.AffExpr) = true
is_aff_or_quad(func::JuMP.QuadExpr) = true
is_aff_or_quad(func) = false

check_model(model) = begin
    !(JuMP.backend(model) isa DiffOpt.Optimizer) && (
        error("Backend must be a DiffOpt.Optimizer. Got: $(JuMP.backend(model))")
    )
    !JuMP.is_solved_and_feasible(model) && (
        error("Model must be solved and feasible. Got: $(JuMP.termination_status(model))")
    )
end

end # module AffineRegions
