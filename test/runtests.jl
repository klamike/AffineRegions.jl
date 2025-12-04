using AffineRegions
using Test

using JuMP, DiffOpt, HiGHS

@testset "Linear" begin
    @testset "RHS" begin
        @testset "Simple" begin
            model = DiffOpt.nonlinear_diff_model(HiGHS.Optimizer)  # FIXME: need ForwardConstraintDual
            set_silent(model)

            p_val = 4.0
            p2_val = 2.0
            @variable(model, x)
            @variable(model, p1 in Parameter(p_val))
            @variable(model, p2 in Parameter(p2_val))
            @constraint(model, cons, x >= 3 * p1)
            @constraint(model, cons2, 4x >= p2)
            @objective(model, Min, 2x)
            optimize!(model)

            c, pl, dl = affine_region(model)

            @test pl[x] == 3p1
            @test dl[cons] == 2
            @test dl[cons2] == 0
            @test length(c) == 1
            @test c[1].func == 12p1 - p2
            @test c[1].set isa MOI.GreaterThan
            @test c[1].set.lower == 0
            
            # should get same region
            set_parameter_value(p1, 1.0)
            set_parameter_value(p2, 5.0)
            optimize!(model)

            c2, pl2, dl2 = affine_region(model)
            @test c[1].func == c2[1].func
            @test c[1].set == c2[1].set
            @test pl == pl2
            @test dl == dl2
            @test value(pl[x]) == value(x)
            @test value(dl[cons]) == dual(cons)

            # should get new region
            set_parameter_value(p1, 1.0)
            set_parameter_value(p2, 13.0)
            optimize!(model)

            c3, pl3, dl3 = affine_region(model)

            @test pl3[x] == p2/4
            @test dl3[cons] == 0
            @test dl3[cons2] == 1/2
            @test length(c3) == 1
            @test c3[1].func == -3p1 + p2/4
            @test c3[1].set isa MOI.GreaterThan
            @test c3[1].set.lower == 0
            @test value(pl[x]) != value(x)
            @test value(dl[cons]) != dual(cons)
        end
    end

    @testset "Obj" begin
        @testset "Simple" begin
            model = DiffOpt.nonlinear_diff_model(HiGHS.Optimizer)  # FIXME: need ForwardConstraintDual
            set_silent(model)

            @variable(model, x ≥ 1)
            @variable(model, y ≥ 2)
            @variable(model, p in Parameter(4.0))
            @constraint(model, cons, x + y == 10)
            @objective(model, Min, p * x + y)
            optimize!(model)

            c, pl, dl = affine_region(model)

            @test pl[x] == 1
            @test pl[y] == 9
            @test dl[cons] == 1
            @test dl[LowerBoundRef(x)] == p - 1
            @test dl[LowerBoundRef(y)] == 0
            @test length(c) == 1
            @test c[1].func == p - 1
            @test c[1].set isa MOI.GreaterThan
            @test c[1].set.lower == 0.0
            
            # should get same region
            set_parameter_value(p, 2.0)
            optimize!(model)

            c2, pl2, dl2 = affine_region(model)

            @test pl2[x] == 1
            @test pl2[y] == 9
            @test dl2[cons] == 1
            @test dl2[LowerBoundRef(x)] == p - 1
            @test dl2[LowerBoundRef(y)] == 0
            @test length(c2) == 1
            @test c2[1].func == p - 1
            @test c2[1].set isa MOI.GreaterThan
            @test c2[1].set.lower == 0.0
            @test value(pl[x]) == value(x)
            @test value(dl[cons]) == dual(cons)
            @test value(dl[LowerBoundRef(x)]) == dual(LowerBoundRef(x))
            @test value(dl[LowerBoundRef(y)]) == dual(LowerBoundRef(y))

            # should get new region
            set_parameter_value(p, -2.0)
            optimize!(model)

            c3, pl3, dl3 = affine_region(model)

            @test pl3[x] == 8
            @test pl3[y] == 2
            @test dl3[cons] == p+0
            @test dl3[LowerBoundRef(x)] == 0
            @test dl3[LowerBoundRef(y)] == -(p - 1)
            @test length(c2) == 1
            @test c3[1].func == -(p - 1)
            @test c3[1].set isa MOI.GreaterThan
            @test c3[1].set.lower == 0.0
            @test value(pl[x]) != value(x)
            @test value(dl[cons]) != dual(cons)
            @test value(dl[LowerBoundRef(x)]) != dual(LowerBoundRef(x))
            @test value(dl[LowerBoundRef(y)]) != dual(LowerBoundRef(y))
        end
    end

    @testset "RHS+Obj" begin
        @testset "Simple" begin
            model = DiffOpt.nonlinear_diff_model(HiGHS.Optimizer)  # FIXME: need ForwardConstraintDual
            set_silent(model)

            @variable(model, x ≥ 1)
            @variable(model, y ≥ 2)
            @variable(model, p in Parameter(4.0))
            @variable(model, p2 in Parameter(8.0))
            @constraint(model, cons, x + y == p2)
            @objective(model, Min, p * x + y)
            optimize!(model)

            c, pl, dl = affine_region(model)

            @test pl[x] == 1
            @test pl[y] == p2 - 1
            @test dl[cons] == 1
            @test dl[LowerBoundRef(x)] == p - 1
            @test dl[LowerBoundRef(y)] == 0
            @test length(c) == 2
            @test c[1].func == p2 - 1
            @test c[1].set isa MOI.GreaterThan
            @test c[1].set.lower == 2.0
            @test c[2].func == p - 1
            @test c[2].set isa MOI.GreaterThan
            @test c[2].set.lower == 0.0

            # should get same region
            set_parameter_value(p, 2.0)
            set_parameter_value(p2, 5.0)
            optimize!(model)

            c2, pl2, dl2 = affine_region(model)

            @test pl2[x] == 1
            @test pl2[y] == p2 - 1
            @test dl2[cons] == 1
            @test dl2[LowerBoundRef(x)] == p - 1
            @test dl2[LowerBoundRef(y)] == 0
            @test length(c2) == 2
            @test c2[1].func == p2 - 1
            @test c2[1].set isa MOI.GreaterThan
            @test c2[1].set.lower == 2.0
            @test c2[2].func == p - 1
            @test c2[2].set isa MOI.GreaterThan
            @test c2[2].set.lower == 0.0

            # should get new region
            set_parameter_value(p, 0.0)
            optimize!(model)
            
            c3, pl3, dl3 = affine_region(model)

            @test pl3[x] == p2 - 2
            @test pl3[y] == 2
            @test dl3[cons] == p+0
            @test dl3[LowerBoundRef(x)] == 0
            @test dl3[LowerBoundRef(y)] == -(p - 1)
            @test length(c3) == 2
            @test c3[1].func == p2 - 2
            @test c3[1].set isa MOI.GreaterThan
            @test c3[1].set.lower == 1.0
            @test c3[2].func == -(p - 1)
            @test c3[2].set isa MOI.GreaterThan
            @test c3[2].set.lower == 0.0
        end
    end
end
