using AffineRegions
using JuMP, DiffOpt, HiGHS
import ParametricOptInterface as POI
import Random
import SparseArrays

function build_same_region_models(n, m, seed)
    Random.seed!(seed)
    A = abs.(randn(m, n)) .+ 0.1
    c = rand(n) .+ 1.0

    ar = DiffOpt.basis_diff_model(HiGHS.Optimizer)
    set_silent(ar)
    @variable(ar, p in Parameter(0.0))
    @variable(ar, x[1:n] >= 0)
    for i in 1:m
        @constraint(ar, sum(A[i, j] * x[j] for j in 1:n) >= 1 + (i == 1 ? p : 0.0))
    end
    @objective(ar, Min, sum(c[j] * x[j] for j in 1:n))

    hi = JuMP.Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(hi)
    @variable(hi, q in Parameter(0.0))
    @variable(hi, y[1:n] >= 0)
    for i in 1:m
        @constraint(hi, sum(A[i, j] * y[j] for j in 1:n) >= 1 + (i == 1 ? q : 0.0))
    end
    @objective(hi, Min, sum(c[j] * y[j] for j in 1:n))
    return ar, hi
end

function direct_batch_benchmark(; n=300, m=120, n_batch=10_000, seed=1)
    ar, hi = build_same_region_models(n, m, seed)
    optimize!(ar)
    optimize!(hi)

    cert = certificate(ar)
    pts = reshape(0.0001 .* randn(n_batch), 1, :)
    @assert all(contains_batch(cert, pts))

    t_direct = @elapsed begin
        mask = contains_batch(cert, pts)
        X = primal_batch(cert, pts)
        @assert all(mask)
        @assert size(X, 2) == n_batch
    end

    p = ar[:p]
    q = hi[:q]
    t_solver = @elapsed begin
        for v in vec(pts)
            set_parameter_value(q, v)
            optimize!(hi)
        end
    end

    return (; n, m, n_batch, t_direct, t_solver, speedup=t_solver / t_direct)
end

function build_piecewise_models(n_lines, seed)
    Random.seed!(seed)
    slopes = sort!(3.0 .* rand(n_lines) .- 1.5)
    intercepts = rand(n_lines) .+ range(0.1, 2.0; length=n_lines)

    ar = DiffOpt.basis_diff_model(HiGHS.Optimizer)
    set_silent(ar)
    @variable(ar, p in Parameter(0.0))
    @variable(ar, x >= 0)
    for i in 1:n_lines
        @constraint(ar, x >= intercepts[i] + slopes[i] * p)
    end
    @objective(ar, Min, x)

    hi = JuMP.Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(hi)
    @variable(hi, q in Parameter(0.0))
    @variable(hi, y >= 0)
    for i in 1:n_lines
        @constraint(hi, y >= intercepts[i] + slopes[i] * q)
    end
    @objective(hi, Min, y)

    return ar, hi
end

function build_piecewise_optimizer_model(n_lines, seed)
    Random.seed!(seed)
    slopes = sort!(3.0 .* rand(n_lines) .- 1.5)
    intercepts = rand(n_lines) .+ range(0.1, 2.0; length=n_lines)

    fast = AffineRegions.Model(HiGHS.Optimizer)
    set_silent(fast)
    @variable(fast, p in Parameter(0.0))
    @variable(fast, x >= 0)
    for i in 1:n_lines
        @constraint(fast, x >= intercepts[i] + slopes[i] * p)
    end
    @objective(fast, Min, x)

    hi = JuMP.Model(() -> POI.Optimizer(HiGHS.Optimizer()))
    set_silent(hi)
    @variable(hi, q in Parameter(0.0))
    @variable(hi, y >= 0)
    for i in 1:n_lines
        @constraint(hi, y >= intercepts[i] + slopes[i] * q)
    end
    @objective(hi, Min, y)

    return fast, hi
end

function build_tangent_optimizer_model(n_lines)
    centers = collect(range(-2.0, 2.0; length=n_lines))
    model = AffineRegions.Model(HiGHS.Optimizer; use_basis_hints=true)
    set_silent(model)
    @variable(model, p in Parameter(centers[1]))
    @variable(model, x >= 0)
    for t in centers
        @constraint(model, x >= 2.0 * t * p - t^2)
    end
    @objective(model, Min, x)
    return centers, model, p, x
end

function build_piecewise_atlas(model; samples=257, lo=-2.0, hi=2.0)
    p = model[:p]
    atlas = AffineRegions.RegionAtlas()
    seen = Set{String}()
    for value in range(lo, hi; length=samples)
        set_parameter_value(p, value)
        optimize!(model)
        cert = certificate(model)
        key = join(cert.basis_cols, ',')
        key in seen && continue
        push!(seen, key)
        add_certificate!(atlas, cert)
    end
    atlas.last_hit = isempty(atlas.certificates) ? 0 : 1
    return atlas
end

function packed_atlas_benchmark(; n_lines=24, n_batch=50_000, seed=2)
    ar, hi = build_piecewise_models(n_lines, seed)
    atlas = build_piecewise_atlas(ar)
    packed = pack(atlas)
    pts = reshape(4.0 .* rand(n_batch) .- 2.0, 1, :)
    workspace = PackedAtlasWorkspace(packed, n_batch)

    ids = query_batch(atlas, pts; workspace)
    @assert all(ids .> 0)

    t_direct = @elapsed begin
        ids = query_batch(atlas, pts; workspace)
        X = primal_batch(atlas, ids, pts; workspace)
        @assert all(ids .> 0)
        @assert size(X) == (1, n_batch)
    end

    q = hi[:q]
    t_solver = @elapsed begin
        for v in vec(pts)
            set_parameter_value(q, v)
            optimize!(hi)
        end
    end

    return (; n_lines, n_cert=length(atlas.certificates), n_batch, t_direct, t_solver, speedup=t_solver / t_direct)
end

function gpu_pack_benchmark(; n_lines=24, n_batch=50_000, seed=2)
    ar, _ = build_piecewise_models(n_lines, seed)
    atlas = build_piecewise_atlas(ar)
    gpu = gpu_pack(atlas; T=Float32, I=Int32)
    pts = reshape(Float32.(4.0 .* rand(n_batch) .- 2.0), 1, :)
    ids = query_batch(gpu, pts)
    @assert all(ids .> 0)

    t_direct = @elapsed begin
        ids = query_batch(gpu, pts)
        X = primal_batch(gpu, ids, pts)
        @assert size(X) == (1, n_batch)
    end

    return (; n_lines, n_cert=length(atlas.certificates), n_batch, t_direct)
end

function sparse_degenerate_build_benchmark(; n=400, m=220, k=6, samples=200, seed=4)
    Random.seed!(seed)
    density = 0.04
    A = SparseArrays.sprand(m, n, density)
    A .= abs.(A) .+ 0.05
    F = randn(m, k)
    c = rand(n) .+ 0.5
    H = randn(n, k) .* 0.1

    model = DiffOpt.basis_diff_model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, p[1:k] in Parameter(0.0))
    @variable(model, x[1:n] >= 0)
    for i in 1:m
        rhs_terms = 1.0 + sum(F[i, j] * p[j] for j in 1:k)
        expr = sum(A[i, q] * x[q] for q in 1:n)
        @constraint(model, expr >= rhs_terms)
        if i <= 40
            @constraint(model, expr >= rhs_terms)
        end
    end
    @objective(model, Min, sum((c[j] + sum(H[j, q] * p[q] for q in 1:k)) * x[j] for j in 1:n))

    atlas = AffineRegions.RegionAtlas()
    t_build = @elapsed begin
        for _ in 1:samples
            vals = 0.02 .* randn(k)
            for j in 1:k
                set_parameter_value(p[j], vals[j])
            end
            optimize!(model)
            add_certificate!(atlas, certificate(model))
        end
    end

    total_rows = sum(size(cert.C, 1) for cert in atlas.certificates)
    shared_rows = length(atlas.constraint_store.c)
    stats = atlas_stats(atlas)
    return (
        ;
        n,
        m,
        k,
        samples,
        n_cert=length(atlas.certificates),
        primal_clusters=stats.n_primal_clusters,
        region_representatives=stats.n_region_representatives,
        total_rows,
        shared_rows,
        compression=shared_rows / max(total_rows, 1),
        t_build,
    )
end

function optimizer_path_benchmark(; n_lines=24, n_batch=20_000, seed=3)
    fast, hi = build_piecewise_optimizer_model(n_lines, seed)
    p = fast[:p]
    x = fast[:x]
    q = hi[:q]
    y = hi[:y]

    for param_value in range(-2.0, 2.0; length=257)
        set_parameter_value(p, param_value)
        optimize!(fast)
    end

    pts = 4.0 .* rand(n_batch) .- 2.0
    opt = JuMP.backend(fast)::AffineRegions.Optimizer
    cached_hits = 0

    t_fast = @elapsed begin
        total = 0.0
        for param_value in pts
            set_parameter_value(p, param_value)
            optimize!(fast)
            cached_hits += opt.use_cached ? 1 : 0
            total += value(x)
        end
        @assert total >= 0.0
    end

    t_solver = @elapsed begin
        total = 0.0
        for param_value in pts
            set_parameter_value(q, param_value)
            optimize!(hi)
            total += value(y)
        end
        @assert total >= 0.0
    end

    return (; n_lines, n_cert=length(opt.atlas.certificates), n_batch, cached_hits, t_fast, t_solver, speedup=t_solver / t_fast)
end

function optimizer_miss_benchmark(; n_lines=64)
    centers, warm_model, warm_p, warm_x = build_tangent_optimizer_model(n_lines)
    _, cold_model, cold_p, cold_x = build_tangent_optimizer_model(n_lines)

    warm_opt = JuMP.backend(warm_model)::AffineRegions.Optimizer
    cold_opt = JuMP.backend(cold_model)::AffineRegions.Optimizer
    cold_opt.use_basis_hints = false

    seed_vals = (first(centers), last(centers))
    for param_value in seed_vals
        set_parameter_value(warm_p, param_value)
        optimize!(warm_model)
        set_parameter_value(cold_p, param_value)
        optimize!(cold_model)
    end

    query_vals = centers[2:(end - 1)]

    t_warm = @elapsed begin
        total = 0.0
        for param_value in query_vals
            set_parameter_value(warm_p, param_value)
            optimize!(warm_model)
            total += value(warm_x)
        end
        @assert total >= 0.0
    end

    t_cold = @elapsed begin
        total = 0.0
        for param_value in query_vals
            set_parameter_value(cold_p, param_value)
            optimize!(cold_model)
            total += value(cold_x)
        end
        @assert total >= 0.0
    end

    return (
        ;
        n_lines,
        warm_hints=warm_opt.basis_hint_applied,
        warm_attempts=warm_opt.basis_hint_attempts,
        cold_hints=cold_opt.basis_hint_applied,
        warm_cert=length(warm_opt.atlas.certificates),
        cold_cert=length(cold_opt.atlas.certificates),
        t_warm,
        t_cold,
        speedup=t_cold / t_warm,
    )
end

function degenerate_certificate_check(; seed=1)
    Random.seed!(seed)
    model = DiffOpt.basis_diff_model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, p in Parameter(0.0))
    @variable(model, 0 <= x[1:20] <= 1)
    @constraint(model, sum(x) == 10 + p)
    @constraint(model, sum(x) == 10 + p)
    @objective(model, Min, sum((i <= 10 ? 1.0 : 1.0) * x[i] for i in 1:20))
    optimize!(model)
    cert = certificate(model)
    return (; rows=size(cert.C, 1), contains=contains_point(cert, [0.0]))
end

function main()
    r = direct_batch_benchmark()
    println("Direct batch benchmark")
    println(r)
    println()
    packed = packed_atlas_benchmark()
    println("Packed atlas benchmark")
    println(packed)
    println()
    gpu = gpu_pack_benchmark()
    println("GPU-ready atlas benchmark")
    println(gpu)
    println()
    build = sparse_degenerate_build_benchmark()
    println("Sparse degenerate build benchmark")
    println(build)
    println()
    wrapper = optimizer_path_benchmark()
    println("Optimizer path benchmark")
    println(wrapper)
    println()
    miss = optimizer_miss_benchmark()
    println("Optional basis-hint miss benchmark")
    println(miss)
    println()
    println("Degenerate certificate check")
    println(degenerate_certificate_check())
end

main()
