using Random
using JuMP
using HiGHS
using ParametricOptInterface
using PowerModels

const POI = ParametricOptInterface
const MOI = JuMP.MOI

PowerModels.silence()

include(joinpath(@__DIR__, "dcopf_89_pegase.jl"))

function build_poi_model(data::DcopfLoadCase)
    model = JuMP.direct_model(POI.Optimizer(HiGHS.Optimizer()))
    JuMP.set_silent(model)

    @variable(model, alpha[g in 1:data.n_groups] in Parameter(1.0))
    @variable(model, va[1:data.N])
    @variable(model, pg[1:data.G])
    @variable(model, pf[1:data.E])

    set_lower_bound.(pg, data.gen_status .* data.pgmin)
    set_upper_bound.(pg, data.gen_status .* data.pgmax)
    set_lower_bound.(pf, data.branch_status .* .-data.smax)
    set_upper_bound.(pf, data.branch_status .* data.smax)

    @expression(model, pd[l in 1:data.L], data.pd_ref[l] * alpha[data.load_group[l]])
    @expression(model, pt[e in 1:data.E], -pf[e])

    @constraint(model, slack_bus, va[data.ref_bus] == 0.0)
    @constraint(
        model,
        kcl_p[i in 1:data.N],
        sum(data.gen_status[g] * pg[g] for g in data.bus_gens[i]) -
        sum(data.branch_status[e] * pf[e] for e in data.bus_arcs_fr[i]) -
        sum(data.branch_status[e] * pt[e] for e in data.bus_arcs_to[i]) ==
        sum(pd[l] for l in data.bus_loads[i]) + data.gs[i]
    )
    @constraint(
        model,
        ohm_pf[e in 1:data.E],
        data.branch_status[e] * (-data.b[e] * (va[data.bus_fr[e]] - va[data.bus_to[e]])) - pf[e] == 0
    )
    @constraint(
        model,
        va_diff_lo[e in 1:data.E],
        data.dvamin[e] <= data.branch_status[e] * (va[data.bus_fr[e]] - va[data.bus_to[e]])
    )
    @constraint(
        model,
        va_diff_hi[e in 1:data.E],
        data.branch_status[e] * (va[data.bus_fr[e]] - va[data.bus_to[e]]) <= data.dvamax[e]
    )

    @objective(model, Min, sum(data.c1[g] * pg[g] for g in 1:data.G if data.gen_status[g]))
    return model, alpha
end

function benchmark_serving(;
    n_train=500,
    n_test=2_000,
    rel_width=0.20,
    sample_seed=123,
    atlas_seed=17,
    batch_size=1_024,
    verbose=true,
)
    if verbose
        println("Starting atlas build: n_train=$(n_train), rel_width=$(rel_width)")
        flush(stdout)
    end
    t_build = @elapsed report = run_dcopf_89_pegase_example(; n_train, n_test=0, verify_test=false, seed=atlas_seed)
    if verbose
        println("Atlas build complete: atlas_size=$(length(report.atlas.certificates)), build_time_s=$(t_build)")
        flush(stdout)
    end
    frozen = report.atlas
    data = report.data

    if verbose
        println("Building POI+HiGHS model")
        flush(stdout)
    end
    poi_model, poi_alpha = build_poi_model(data)
    if verbose
        println("POI+HiGHS model ready")
        flush(stdout)
    end
    rng = MersenneTwister(sample_seed)

    if verbose
        println("Generating feasible POI sample set")
        flush(stdout)
    end
    t_sample = @elapsed samples, attempts = sample_feasible_group_load_scenarios(poi_model, poi_alpha, data, n_test; rng, rel_width)
    if verbose
        println("Sample generation complete: attempts=$(attempts), sample_time_s=$(t_sample)")
        flush(stdout)
    end

    if verbose
        println("Running warmup")
        flush(stdout)
    end
    ids0 = query_batch(frozen, samples)
    covered0 = findall(!iszero, ids0)
    isempty(covered0) || primal_batch(frozen, ids0[covered0], samples[:, covered0])
    AffineRegions._set_param_vector!(poi_alpha, view(samples, :, 1))
    optimize!(poi_model)
    if verbose
        println("Warmup complete")
        flush(stdout)
    end

    GC.gc()
    if verbose
        println("Timing atlas query + primal reconstruction in chunks of $(batch_size)")
        flush(stdout)
    end
    t_atlas = @elapsed begin
        for first_col in 1:batch_size:n_test
            last_col = min(first_col + batch_size - 1, n_test)
            cols = first_col:last_col
            ids = query_batch(frozen, samples[:, cols])
            covered = findall(!iszero, ids)
            isempty(covered) || primal_batch(frozen, ids[covered], samples[:, cols[covered]])
        end
    end
    if verbose
        println("Atlas timing complete: atlas_time_s=$(t_atlas)")
        flush(stdout)
    end

    ids = query_batch(frozen, samples)
    covered = findall(!iszero, ids)
    misses = findall(iszero, ids)
    if verbose
        println("Coverage split: covered=$(length(covered)), misses=$(length(misses))")
        flush(stdout)
    end

    GC.gc()
    if verbose
        println("Timing POI+HiGHS on all test points")
        flush(stdout)
    end
    t_poi_all = @elapsed begin
        for j in 1:n_test
            AffineRegions._set_param_vector!(poi_alpha, view(samples, :, j))
            optimize!(poi_model)
            @assert termination_status(poi_model) == MOI.OPTIMAL
        end
    end
    if verbose
        println("POI+HiGHS all-points timing complete: solve_time_s=$(t_poi_all)")
        flush(stdout)
    end

    GC.gc()
    if verbose
        println("Timing POI+HiGHS on covered points only")
        flush(stdout)
    end
    t_poi_covered = @elapsed begin
        for j in covered
            AffineRegions._set_param_vector!(poi_alpha, view(samples, :, j))
            optimize!(poi_model)
            @assert termination_status(poi_model) == MOI.OPTIMAL
        end
    end
    if verbose
        println("POI+HiGHS covered-points timing complete: covered_time_s=$(t_poi_covered)")
        flush(stdout)
    end

    GC.gc()
    if verbose
        println("Timing POI+HiGHS miss fallback only")
        flush(stdout)
    end
    t_poi_misses = @elapsed begin
        for j in misses
            AffineRegions._set_param_vector!(poi_alpha, view(samples, :, j))
            optimize!(poi_model)
            @assert termination_status(poi_model) == MOI.OPTIMAL
        end
    end
    if verbose
        println("POI+HiGHS miss timing complete: miss_time_s=$(t_poi_misses)")
        flush(stdout)
    end

    return (
        n_train=n_train,
        n_test=n_test,
        batch_size=batch_size,
        feasible_sample_attempts=attempts,
        atlas_size=length(frozen.certificates),
        covered_points=length(covered),
        miss_points=length(misses),
        coverage=length(covered) / n_test,
        atlas_query_plus_primal_s=t_atlas,
        poi_highs_all_solve_s=t_poi_all,
        poi_highs_covered_solve_s=t_poi_covered,
        blended_atlas_plus_miss_fallback_s=t_atlas + t_poi_misses,
        hit_only_speedup=t_poi_covered / t_atlas,
        blended_speedup=t_poi_all / (t_atlas + t_poi_misses),
    )
end

function main()
    result = benchmark_serving()
    println("PEGASE serving benchmark")
    println("  n_train                        = ", result.n_train)
    println("  n_test                         = ", result.n_test)
    println("  batch_size                     = ", result.batch_size)
    println("  feasible_sample_attempts       = ", result.feasible_sample_attempts)
    println("  atlas_size                     = ", result.atlas_size)
    println("  covered_points                 = ", result.covered_points)
    println("  miss_points                    = ", result.miss_points)
    println("  coverage                       = ", result.coverage)
    println("  atlas_query_plus_primal_s      = ", result.atlas_query_plus_primal_s)
    println("  poi_highs_all_solve_s          = ", result.poi_highs_all_solve_s)
    println("  poi_highs_covered_solve_s      = ", result.poi_highs_covered_solve_s)
    println("  blended_atlas_plus_fallback_s  = ", result.blended_atlas_plus_miss_fallback_s)
    println("  hit_only_speedup               = ", result.hit_only_speedup)
    println("  blended_speedup                = ", result.blended_speedup)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
