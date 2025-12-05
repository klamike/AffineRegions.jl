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
                  #    in some cases, this also implies optimality

    primal_law,   # Dict{VariableRef,AffExpr} where each AffExpr
                  #    maps parameters to primal solution

    dual_law      # Dict{ConstraintRef,AffExpr} where each AffExpr
                  #    maps parameters to dual solution
) = affine_region(model)
```
