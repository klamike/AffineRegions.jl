using AffineRegions
using Test

using JuMP, DiffOpt, HiGHS
using Random
using Serialization

const MOI = JuMP.MOI

function basis_model()
    model = DiffOpt.basis_diff_model(HiGHS.Optimizer)
    set_silent(model)
    return model
end

@testset "Certificate Basics" begin
    @testset "RHS-only basis region" begin
        model = basis_model()
        @variable(model, x >= 0)
        @variable(model, p1 in Parameter(4.0))
        @variable(model, p2 in Parameter(2.0))
        @constraint(model, c1, x >= 3p1)
        @constraint(model, c2, 4x >= p2)
        @objective(model, Min, 2x)
        optimize!(model)

        lp = extract_lp(model)
        @test lp.rhs_parametric
        @test !lp.obj_parametric
        cert = certificate(model)
        @test cert.P[1, 1] ≈ 3.0 atol = 1e-8
        @test cert.P[1, 2] ≈ 0.0 atol = 1e-8
        @test cert.p0[1] ≈ 0.0 atol = 1e-8
        @test all(abs.(cert.D) .<= 1e-8)

        p = [parameter_value(p1), parameter_value(p2)]
        @test all(cert.C * p .+ cert.c .>= -1e-8)

        set_parameter_value(p1, 1.0)
        set_parameter_value(p2, 5.0)
        optimize!(model)
        p_same = [parameter_value(p1), parameter_value(p2)]
        @test contains_point(cert, p_same)

        set_parameter_value(p1, 1.0)
        set_parameter_value(p2, 13.0)
        optimize!(model)
        p_other = [parameter_value(p1), parameter_value(p2)]
        @test !contains_point(cert, p_other)
        @test contains_batch(cert, hcat(p, p_same)) == [true, true]
    end

    @testset "Equality, free variable, and upper bound" begin
        model = basis_model()
        @variable(model, x)
        @variable(model, 0 <= y <= 3)
        @variable(model, p in Parameter(2.0))
        @constraint(model, eq, x + y == 4)
        @objective(model, Min, p * x + y)
        optimize!(model)

        lp = extract_lp(model)
        @test !lp.rhs_parametric
        @test lp.obj_parametric
        @test size(lp.A, 1) == 2
        @test size(lp.A, 2) == 4

        cert = certificate(model)
        @test value(x) ≈ 1.0 atol = 1e-8
        @test value(y) ≈ 3.0 atol = 1e-8
        @test cert.P[1, 1] ≈ 0.0 atol = 1e-8
        @test cert.P[2, 1] ≈ 0.0 atol = 1e-8
        @test cert.p0[1] ≈ 1.0 atol = 1e-8
        @test cert.p0[2] ≈ 3.0 atol = 1e-8
        @test all(abs.(cert.P) .<= 1e-8)
        @test all(cert.C * [2.0] .+ cert.c .>= -1e-8)
        @test any(cert.C * [0.0] .+ cert.c .< -1e-8)

        region = affine_region(model)
        @test value(region.primal_law[x]) ≈ value(x) atol = 1e-8
        @test value(region.primal_law[y]) ≈ value(y) atol = 1e-8
        @test haskey(region.dual_law, eq)
        @test haskey(region.dual_law, UpperBoundRef(y))
        @test haskey(region.dual_law, LowerBoundRef(y))
    end

    @testset "Max sense" begin
        model = basis_model()
        @variable(model, x >= 0)
        @variable(model, p in Parameter(2.0))
        @constraint(model, x <= p)
        @objective(model, Max, 3x)
        optimize!(model)

        cert = certificate(model)
        @test primal_solution(cert, [2.0])[1] ≈ 2.0 atol = 1e-8
        @test dual_solution(cert, [2.0])[1] ≈ -3.0 atol = 1e-8
        @test contains_point(cert, [2.0])
        @test objective_value(model) ≈ 6.0 atol = 1e-8
    end
end

@testset "Degeneracy" begin
    @testset "Primal degeneracy duplicate equality" begin
        model = basis_model()
        @variable(model, p in Parameter(0.0))
        @variable(model, 0 <= x[1:20] <= 1)
        @constraint(model, eq1, sum(x) == 10 + p)
        @constraint(model, eq2, sum(x) == 10 + p)
        @objective(model, Min, sum(x))
        optimize!(model)

        cert = certificate(model)
        @test contains_point(cert, [0.0])
    end

    @testset "Dual degeneracy zero reduced costs" begin
        model = basis_model()
        @variable(model, p in Parameter(0.0))
        @variable(model, x[1:4] >= 0)
        @constraint(model, sum(x) == 1.0)
        @objective(model, Min, p * x[1] + p * x[2] + x[3] + x[4])
        optimize!(model)

        cert = certificate(model)
        @test contains_point(cert, [0.0])
        vals = primal_batch(cert, reshape([0.0, 0.0], 1, :))
        @test size(vals) == (4, 2)
    end
end

@testset "Constraint Reduction" begin
    C = [
        1.0 0.0
        2.0 0.0
        0.0 0.0
        1.0 0.0
    ]
    c = [0.0, 0.0, 1.0, -1.0]
    C2, c2 = AffineRegions._reduce_constraints(C, c; exact=false)
    @test size(C2, 1) == 1
    @test C2[1, 1] ≈ 1.0 atol = 1e-8
    @test c2[1] ≈ -1.0 atol = 1e-8
end

@testset "Optimizer Cache" begin
    model = AffineRegions.Model(HiGHS.Optimizer)
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, con, x >= p)
    @objective(model, Min, x)

    optimize!(model)
    @test is_solved_and_feasible(model)
    opt = JuMP.backend(model)::AffineRegions.Optimizer
    @test length(opt.atlas.certificates) == 1

    set_parameter_value(p, 2.0)
    optimize!(model)
    @test opt.use_cached
    @test value(x) ≈ 2.0 atol = 1e-8
    @test objective_value(model) ≈ 2.0 atol = 1e-8
end

@testset "Optimizer Cache Max Sense" begin
    model = AffineRegions.Model(HiGHS.Optimizer)
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, con, x <= p)
    @objective(model, Max, 2x)

    optimize!(model)
    @test is_solved_and_feasible(model)
    opt = JuMP.backend(model)::AffineRegions.Optimizer
    @test length(opt.atlas.certificates) == 1

    set_parameter_value(p, 2.0)
    optimize!(model)
    @test opt.use_cached
    @test value(x) ≈ 2.0 atol = 1e-8
    @test objective_value(model) ≈ 4.0 atol = 1e-8
end

@testset "Optimizer Multi-Region Cache" begin
    model = AffineRegions.Model(HiGHS.Optimizer)
    @variable(model, x >= 0)
    @variable(model, p in Parameter(0.0))
    @constraint(model, c1, x >= p)
    @constraint(model, c2, x >= -p)
    @objective(model, Min, x)

    set_parameter_value(p, -1.0)
    optimize!(model)
    set_parameter_value(p, 1.0)
    optimize!(model)

    opt = JuMP.backend(model)::AffineRegions.Optimizer
    @test length(opt.atlas.certificates) == 2
    packed = pack(opt.atlas)
    @test packed === pack(opt.atlas)

    set_parameter_value(p, -0.5)
    optimize!(model)
    @test opt.use_cached
    @test value(x) ≈ 0.5 atol = 1e-8

    set_parameter_value(p, 0.5)
    optimize!(model)
    @test opt.use_cached
    @test value(x) ≈ 0.5 atol = 1e-8
end

@testset "Optimizer Basis Hints" begin
    model = AffineRegions.Model(HiGHS.Optimizer; use_basis_hints=true)
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p in Parameter(-2.0))
    @constraint(model, c1, x >= p)
    @constraint(model, c2, x >= -p)
    @constraint(model, c3, x + y >= 1.0)
    @objective(model, Min, x + 0.1y)

    optimize!(model)
    set_parameter_value(p, 2.0)
    optimize!(model)

    opt = JuMP.backend(model)::AffineRegions.Optimizer
    @test length(opt.atlas.certificates) == 2

    set_parameter_value(p, 0.2)
    optimize!(model)
    @test !opt.use_cached
    @test opt.basis_hint_attempts >= 1
    @test opt.basis_hint_applied >= 1
    @test length(opt.atlas.certificates) == 3
    @test value(x) ≈ 0.2 atol = 1e-8
    @test value(y) ≈ 0.8 atol = 1e-8
end

@testset "Packed Atlas" begin
    model = basis_model()
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, cpos, x >= p)
    @constraint(model, cneg, x >= -p)
    @objective(model, Min, x)

    optimize!(model)
    cert_pos = certificate(model)

    set_parameter_value(p, -1.0)
    optimize!(model)
    cert_neg = certificate(model)

    atlas = RegionAtlas([cert_pos, cert_neg], 1)
    packed = pack(atlas)
    @test packed === pack(atlas)
    workspace = PackedAtlasWorkspace(packed, 4)
    pts = reshape([-1.0, -0.5, 0.5, 1.0], 1, :)

    hits = contains_batch(atlas, pts; workspace)
    @test size(hits) == (2, 4)

    ids = query_batch(atlas, pts; workspace)
    @test all(ids .> 0)
    @test length(unique(ids)) == 2

    X = primal_batch(atlas, ids, pts; workspace)
    Λ = dual_batch(atlas, ids, pts; workspace)
    for j in 1:size(pts, 2)
        cert = atlas.certificates[ids[j]]
        pcol = view(pts, :, j)
        @test contains_point(cert, pcol)
        @test X[:, j] ≈ primal_solution(cert, pcol) atol = 1e-8
        @test Λ[:, j] ≈ dual_solution(cert, pcol) atol = 1e-8
    end

    add_certificate!(atlas, cert_pos)
    @test length(atlas.certificates) == 2
    @test length(atlas.constraint_store.c) <= sum(size(cert.C, 1) for cert in atlas.certificates)
    @test pack(atlas) === packed
end

@testset "Atlas Clustering" begin
    model = basis_model()
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, x >= p)
    @objective(model, Min, x)

    optimize!(model)
    cert1 = certificate(model)
    set_parameter_value(p, 2.0)
    optimize!(model)
    cert2 = certificate(model)

    atlas = RegionAtlas()
    add_certificate!(atlas, cert1)
    add_certificate!(atlas, cert2)
    stats = atlas_stats(atlas)
    @test stats.n_certificates == 1
    @test stats.n_primal_clusters == 1
    @test stats.n_region_representatives == 1
end

@testset "GPU Pack" begin
    model = basis_model()
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, cpos, x >= p)
    @constraint(model, cneg, x >= -p)
    @objective(model, Min, x)

    optimize!(model)
    cert_pos = certificate(model)
    set_parameter_value(p, -1.0)
    optimize!(model)
    cert_neg = certificate(model)

    atlas = RegionAtlas([cert_pos, cert_neg], 1)
    gpu = gpu_pack(atlas; T=Float32, I=Int32)
    pts = reshape(Float32[-1.0, -0.5, 0.5, 1.0], 1, :)
    ids = query_batch(gpu, pts)
    @test all(ids .> 0)
    X = primal_batch(gpu, ids, pts)
    Λ = dual_batch(gpu, ids, pts)
    frozen = freeze(atlas)
    @test ids == query_batch(frozen, Matrix{Float64}(pts))
    @test Matrix{Float64}(X) ≈ primal_batch(frozen, ids, Matrix{Float64}(pts)) atol = 1e-5
    @test Matrix{Float64}(Λ) ≈ dual_batch(frozen, ids, Matrix{Float64}(pts)) atol = 1e-5
end

@testset "Atlas Lifecycle" begin
    model = basis_model()
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, cpos, x >= p)
    @constraint(model, cneg, x >= -p)
    @objective(model, Min, x)

    optimize!(model)
    cert_pos = certificate(model)
    set_parameter_value(p, -1.0)
    optimize!(model)
    cert_neg = certificate(model)

    atlas = RegionAtlas([cert_pos, cert_neg], 1; policy=NumericalPolicy(1e-10, 1e-8, 1e-8, 1e-9, 1e-8, false, 8, 256))
    stats = atlas_stats(atlas)
    @test stats.n_certificates == 2
    @test stats.shared_region_rows <= stats.total_region_rows

    frozen = freeze(atlas)
    pts = reshape([-0.5, 0.5], 1, :)
    ids = query_batch(frozen, pts)
    @test all(ids .> 0)
    result = solve_batch(frozen, pts)
    @test result.ids == ids
    @test size(result.primal) == (1, 2)
    @test size(result.dual) == (3, 2)

    hits_dest = falses(length(frozen.certificates), size(pts, 2))
    ids_dest = zeros(Int, size(pts, 2))
    primal_dest = zeros(1, size(pts, 2))
    dual_dest = zeros(3, size(pts, 2))
    contains_batch!(hits_dest, frozen, pts)
    query_batch!(ids_dest, frozen, pts)
    primal_batch!(primal_dest, frozen, ids, pts)
    dual_batch!(dual_dest, frozen, ids, pts)
    @test ids_dest == ids
    @test primal_dest ≈ result.primal atol = 1e-8
    @test dual_dest ≈ result.dual atol = 1e-8

    path = tempname()
    save_atlas(path, atlas)
    loaded = load_atlas(path)
    loaded_frozen = load_atlas(path; frozen=true)
    @test atlas_stats(loaded).n_certificates == 2
    @test query_batch(loaded_frozen, pts) == ids

    legacy_path = tempname()
    Serialization.serialize(legacy_path, snapshot(atlas))
    loaded_legacy = load_atlas(legacy_path)
    @test atlas_stats(loaded_legacy).n_certificates == 2
    @test query_batch(freeze(loaded_legacy), pts) == ids

    remove_certificate!(loaded, 1)
    @test length(loaded.certificates) == 1
    compact!(loaded)
    clear!(loaded)
    @test isempty(loaded.certificates)
end

@testset "Frozen Atlas Concurrency" begin
    model = basis_model()
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, cpos, x >= p)
    @constraint(model, cneg, x >= -p)
    @objective(model, Min, x)

    optimize!(model)
    cert_pos = certificate(model)
    set_parameter_value(p, -1.0)
    optimize!(model)
    cert_neg = certificate(model)

    frozen = freeze(RegionAtlas([cert_pos, cert_neg], 1))
    pts = reshape(range(-1.0, 1.0; length=201), 1, :)
    ids_ref = query_batch(frozen, pts)
    primal_ref = primal_batch(frozen, ids_ref, pts)
    dual_ref = dual_batch(frozen, ids_ref, pts)

    n_jobs = max(1, Threads.nthreads() * 4)
    ids_jobs = [zeros(Int, length(ids_ref)) for _ in 1:n_jobs]
    primal_jobs = [zeros(size(primal_ref)) for _ in 1:n_jobs]
    dual_jobs = [zeros(size(dual_ref)) for _ in 1:n_jobs]
    Threads.@threads for job in 1:n_jobs
        ids_jobs[job] .= query_batch(frozen, pts)
        primal_jobs[job] .= primal_batch(frozen, ids_jobs[job], pts)
        dual_jobs[job] .= dual_batch(frozen, ids_jobs[job], pts)
    end
    for job in 1:n_jobs
        @test ids_jobs[job] == ids_ref
        @test primal_jobs[job] ≈ primal_ref atol = 1e-8
        @test dual_jobs[job] ≈ dual_ref atol = 1e-8
    end
end

@testset "Offline Atlas Builder" begin
    model = basis_model()
    @variable(model, x >= 0)
    @variable(model, p in Parameter(0.0))
    @constraint(model, cpos, x >= p)
    @constraint(model, cneg, x >= -p)
    @objective(model, Min, x)

    samples = reshape([-2.0, -1.0, -0.25, 0.25, 1.0, 2.0], 1, :)
    built = build_atlas(model, samples; freeze_result=true)
    frozen = built.atlas
    report = built.report
    @test frozen isa FrozenAtlas
    @test report.attempted_points == size(samples, 2)
    @test report.feasible_points == size(samples, 2)
    @test report.skipped_points == 0
    @test report.final_certificates == 2
    @test all(query_batch(frozen, samples) .> 0)
end

@testset "Atlas Metrics and Policies" begin
    model = basis_model()
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, cpos, x >= p)
    @constraint(model, cneg, x >= -p)
    @objective(model, Min, x)

    optimize!(model)
    cert_pos = certificate(model)
    set_parameter_value(p, -1.0)
    optimize!(model)
    cert_neg = certificate(model)

    atlas = RegionAtlas(; atlas_policy=AtlasPolicy(max_certificates=1, eviction=:fifo))
    add_certificate!(atlas, cert_pos)
    add_certificate!(atlas, cert_neg)
    @test length(atlas.certificates) == 1
    metrics = atlas_metrics(atlas)
    @test metrics.certificates_added == 2
    @test metrics.evictions == 1

    frozen = freeze(RegionAtlas([cert_pos], 1))
    ids = query_batch(frozen, reshape([1.0, -1.0], 1, :))
    frozen_metrics = atlas_metrics(frozen)
    @test ids == [1, 0]
    @test frozen_metrics.batch_query_calls == 1
    @test frozen_metrics.query_calls == 2
    @test frozen_metrics.query_hits == 1
    @test frozen_metrics.query_misses == 1
    reset_metrics!(frozen)
    @test atlas_metrics(frozen).query_calls == 0

    blocked = RegionAtlas(; atlas_policy=AtlasPolicy(learning_enabled=false))
    add_certificate!(blocked, cert_pos)
    @test isempty(blocked.certificates)
    @test atlas_metrics(blocked).policy_rejections == 1
end

@testset "Numerical Stress" begin
    rng = MersenneTwister(7)
    model = basis_model()
    @variable(model, x >= 0)
    @variable(model, y >= 0)
    @variable(model, p1 in Parameter(0.0))
    @variable(model, p2 in Parameter(0.0))
    @constraint(model, c1, x + y >= 1.0 + 1e-6p1)
    @constraint(model, c2, x + (1.0 + 1e-9) * y >= 1.0 + 1e-6p2)
    @constraint(model, c3, x <= 1.0 + 1e-6p1)
    @objective(model, Min, x + (1.0 + 1e-10) * y)

    samples = zeros(2, 24)
    for j in axes(samples, 2)
        samples[:, j] .= 0.5 .* randn(rng, 2)
    end

    built = build_atlas(model, samples; atlas_policy=AtlasPolicy(max_certificates=16))
    atlas = built.atlas
    @test built.report.feasible_points == size(samples, 2)
    @test length(atlas.certificates) <= 16
    for cert in atlas.certificates
        @test contains_point(cert, cert.anchor_p; tol=1e-7)
    end

    frozen = freeze(atlas)
    ids = query_batch(frozen, samples)
    @test all(ids .> 0)
    primal = primal_batch(frozen, ids, samples)

    for j in axes(samples, 2)
        AffineRegions._set_param_vector!(extract_lp(model).parameters, @view samples[:, j])
        optimize!(model)
        @test value(x) ≈ primal[1, j] atol = 1e-6
        @test value(y) ≈ primal[2, j] atol = 1e-6
    end

    path = tempname()
    save_atlas(path, atlas)
    loaded = load_atlas(path; frozen=true)
    @test query_batch(loaded, samples) == ids
    @test loaded.atlas_policy.max_certificates == atlas.atlas_policy.max_certificates
end

@testset "DCOPF Example" begin
    include(joinpath(dirname(@__FILE__), "..", "examples", "dcopf_parametric_load.jl"))
    report = run_dcopf_parametric_load_example(; n_train=20, n_test=0, seed=11)
    @test report.atlas_size > 0
    @test report.train_coverage == 1.0
    @test report.max_pg_error <= 1e-7
    @test report.max_pf_error <= 1e-7
    @test report.max_obj_error <= 1e-7
end

@testset "Optional Extensions" begin
    if Base.find_package("Gurobi") !== nothing
        try
            import Gurobi
            @test Base.get_extension(AffineRegions, :AffineRegionsGurobiExt) !== nothing
        catch
            @test true
        end
    else
        @test true
    end
end

@testset "Failure Modes" begin
    model = basis_model()
    @variable(model, x >= 0)
    @variable(model, p in Parameter(1.0))
    @constraint(model, x^2 >= p)
    @objective(model, Min, x)
    @test_throws UnsupportedModelError extract_lp(model)

    model2 = basis_model()
    @variable(model2, x >= 0)
    @variable(model2, p in Parameter(1.0))
    @constraint(model2, x >= p)
    @objective(model2, Min, x)
    @test_throws InvalidModelStateError certificate(model2)

    model3 = basis_model()
    @variable(model3, x >= 0)
    @variable(model3, p in Parameter(0.0))
    @constraint(model3, x >= p)
    @constraint(model3, x >= -p)
    @objective(model3, Min, x)
    optimize!(model3)
    atlas = RegionAtlas([certificate(model3)])
    frozen = freeze(atlas)
    @test_throws QueryCoverageError solve_batch(frozen, reshape([1.0], 1, :))
end
