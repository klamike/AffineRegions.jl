using AffineRegions
using JuMP, DiffOpt, HiGHS
import ParametricOptInterface as POI

function direct_smoke()
    ar = DiffOpt.basis_diff_model(HiGHS.Optimizer)
    set_silent(ar)
    @variable(ar, p in Parameter(0.0))
    @variable(ar, x[1:40] >= 0)
    for i in 1:20
        @constraint(ar, sum((1.0 + 0.01 * i * j) * x[j] for j in 1:40) >= 1 + (i == 1 ? p : 0.0))
    end
    @objective(ar, Min, sum((1.0 + 0.01 * j) * x[j] for j in 1:40))
    optimize!(ar)
    cert = certificate(ar)
    pts = reshape(0.001 .* collect(-100:99), 1, :)

    hi = JuMP.Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(hi)
    @variable(hi, q in Parameter(0.0))
    @variable(hi, y[1:40] >= 0)
    for i in 1:20
        @constraint(hi, sum((1.0 + 0.01 * i * j) * y[j] for j in 1:40) >= 1 + (i == 1 ? q : 0.0))
    end
    @objective(hi, Min, sum((1.0 + 0.01 * j) * y[j] for j in 1:40))
    optimize!(hi)

    t_direct = @elapsed primal_batch(cert, pts)
    q = hi[:q]
    t_solver = @elapsed for v in vec(pts)
        set_parameter_value(q, v)
        optimize!(hi)
    end
    return t_solver / t_direct
end

function packed_smoke()
    model = DiffOpt.basis_diff_model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, p in Parameter(0.0))
    @variable(model, x >= 0)
    for (a, b) in ((1.0, 0.0), (-1.0, 0.0), (0.5, 1.0), (-0.5, 1.0))
        @constraint(model, x >= b + a * p)
    end
    @objective(model, Min, x)

    atlas = RegionAtlas()
    for v in (-2.0, -0.25, 0.25, 2.0)
        set_parameter_value(p, v)
        optimize!(model)
        add_certificate!(atlas, certificate(model))
    end
    frozen = freeze(atlas)
    pts = reshape(range(-2.0, 2.0; length=1000), 1, :)
    ids = query_batch(frozen, pts)
    result = solve_batch(frozen, pts)
    @assert result.ids == ids

    hi = JuMP.Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(hi)
    @variable(hi, q in Parameter(0.0))
    @variable(hi, y >= 0)
    for (a, b) in ((1.0, 0.0), (-1.0, 0.0), (0.5, 1.0), (-0.5, 1.0))
        @constraint(hi, y >= b + a * q)
    end
    @objective(hi, Min, y)
    optimize!(hi)

    t_direct = @elapsed solve_batch(frozen, pts)
    q = hi[:q]
    t_solver = @elapsed for v in vec(pts)
        set_parameter_value(q, v)
        optimize!(hi)
    end
    return t_solver / t_direct
end

direct_speedup = direct_smoke()
packed_speedup = packed_smoke()

println("perf_smoke: direct_speedup=$(round(direct_speedup; digits=2)) packed_speedup=$(round(packed_speedup; digits=2))")

@assert direct_speedup > 1.2
@assert packed_speedup > 1.2
