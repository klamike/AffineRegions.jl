module AffineRegionsGurobiExt

import AffineRegions
import Gurobi
import JuMP

const MOI = JuMP.MOI

struct GurobiBasisBackend <: AffineRegions.NativeBasisBackend end

AffineRegions._native_basis_backend(opt::Gurobi.Optimizer) = GurobiBasisBackend()
AffineRegions._native_basis_name(::GurobiBasisBackend) = "Gurobi"

function AffineRegions._capture_native_basis_hint(::GurobiBasisBackend, opt, variables, row_constraints)
    var_status = Vector{MOI.BasisStatusCode}(undef, length(variables))
    for (i, vr) in enumerate(variables)
        var_status[i] = MOI.get(opt, MOI.VariableBasisStatus(), JuMP.index(vr))
    end
    row_status = Vector{MOI.BasisStatusCode}(undef, length(row_constraints))
    for (i, cr) in enumerate(row_constraints)
        row_status[i] = MOI.get(opt, MOI.ConstraintBasisStatus(), JuMP.index(cr))
    end
    return AffineRegions.MOIBasisHint(var_status, row_status)
end

function AffineRegions._apply_native_basis_hint!(::GurobiBasisBackend, opt, variables, row_constraints, hint::AffineRegions.MOIBasisHint)
    length(hint.var_status) == length(variables) || return false
    length(hint.row_status) == length(row_constraints) || return false
    try
        for (i, vr) in enumerate(variables)
            MOI.set(opt, MOI.VariableBasisStatus(), JuMP.index(vr), hint.var_status[i])
        end
        for (i, cr) in enumerate(row_constraints)
            MOI.set(opt, MOI.ConstraintBasisStatus(), JuMP.index(cr), hint.row_status[i])
        end
        return true
    catch
        return false
    end
end

end # module AffineRegionsGurobiExt
