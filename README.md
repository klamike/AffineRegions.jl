# AffineRegions.jl

A tiny package for computing "affine regions" given a solved parametric JuMP model. The main API is `affine_region(model)`:


```julia
using JuMP, DiffOpt, HiGHS
model = DiffOpt.diff_model(HiGHS.Optimizer)
# ... parametric model definition ...
optimize!(model)

(
    constraints,  # constraints (in terms of parameters)
                  #    within which the laws are primal-dual feasible;
                  #    if parameters enter the objective and/or RHS linearly,
                  #    the solution given by the laws is also optimal

    primal_law,   # Dict{VariableRef,AffExpr} where each AffExpr
                  #    maps parameters to optimal primal solution

    dual_law      # Dict{ConstraintRef,AffExpr} where each AffExpr
                  #    maps parameters to optimal dual solution
) = affine_region(model)
```
