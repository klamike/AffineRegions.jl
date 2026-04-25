using AffineRegions
using DiffOpt
using HiGHS
using JuMP
using Random

struct DcopfLoadCase
    N::Int
    E::Int
    G::Int
    L::Int
    ref_bus::Int
    bus_fr::Vector{Int}
    bus_to::Vector{Int}
    b::Vector{Float64}
    smax::Vector{Float64}
    dvamin::Vector{Float64}
    dvamax::Vector{Float64}
    gen_bus::Vector{Int}
    pgmin::Vector{Float64}
    pgmax::Vector{Float64}
    c1::Vector{Float64}
    gs::Vector{Float64}
    pd_ref::Vector{Float64}
    bus_arcs_fr::Vector{Vector{Int}}
    bus_arcs_to::Vector{Vector{Int}}
    bus_gens::Vector{Vector{Int}}
    bus_loads::Vector{Vector{Int}}
end

function dcopf_case5()
    N = 5
    bus_fr = [1, 1, 2, 2, 3, 4]
    bus_to = [2, 3, 3, 4, 5, 5]
    b = [10.0, 8.0, 9.0, 7.0, 6.0, 8.0]
    smax = [1.20, 1.10, 0.90, 1.00, 0.90, 0.95]
    dvamin = fill(-0.60, length(bus_fr))
    dvamax = fill(0.60, length(bus_fr))

    gen_bus = [1, 3, 5]
    pgmin = [0.0, 0.0, 0.0]
    pgmax = [1.60, 0.90, 0.70]
    c1 = [10.0, 22.0, 40.0]

    gs = zeros(Float64, N)
    pd_ref = [0.55, 0.45, 0.35, 0.25]

    bus_arcs_fr = [Int[] for _ in 1:N]
    bus_arcs_to = [Int[] for _ in 1:N]
    for e in eachindex(bus_fr)
        push!(bus_arcs_fr[bus_fr[e]], e)
        push!(bus_arcs_to[bus_to[e]], e)
    end

    bus_gens = [Int[] for _ in 1:N]
    for g in eachindex(gen_bus)
        push!(bus_gens[gen_bus[g]], g)
    end

    bus_loads = [Int[] for _ in 1:N]
    for (l, bus) in enumerate(2:5)
        push!(bus_loads[bus], l)
    end

    return DcopfLoadCase(
        N,
        length(bus_fr),
        length(gen_bus),
        length(pd_ref),
        1,
        bus_fr,
        bus_to,
        b,
        smax,
        dvamin,
        dvamax,
        gen_bus,
        pgmin,
        pgmax,
        c1,
        gs,
        pd_ref,
        bus_arcs_fr,
        bus_arcs_to,
        bus_gens,
        bus_loads,
    )
end

function build_parametric_dcopf_model(data::DcopfLoadCase; optimizer=HiGHS.Optimizer)
    model = DiffOpt.basis_diff_model(optimizer)
    JuMP.set_silent(model)

    @variable(model, d[l in 1:data.L] in Parameter(data.pd_ref[l]))
    @variable(model, va[1:data.N])
    @variable(model, pg[1:data.G])
    @variable(model, pf[1:data.E])

    set_lower_bound.(pg, data.pgmin)
    set_upper_bound.(pg, data.pgmax)
    set_lower_bound.(pf, .-data.smax)
    set_upper_bound.(pf, data.smax)

    @constraint(model, slack_bus, va[data.ref_bus] == 0.0)
    @expression(model, pt[e in 1:data.E], -pf[e])

    @constraint(
        model,
        kcl_p[i in 1:data.N],
        sum(pg[g] for g in data.bus_gens[i]) -
        sum(pf[e] for e in data.bus_arcs_fr[i]) -
        sum(pt[e] for e in data.bus_arcs_to[i]) ==
        sum(d[l] for l in data.bus_loads[i]) + data.gs[i]
    )

    @constraint(
        model,
        ohm_pf[e in 1:data.E],
        -data.b[e] * (va[data.bus_fr[e]] - va[data.bus_to[e]]) - pf[e] == 0
    )

    @constraint(
        model,
        va_diff_lo[e in 1:data.E],
        data.dvamin[e] <= va[data.bus_fr[e]] - va[data.bus_to[e]]
    )
    @constraint(
        model,
        va_diff_hi[e in 1:data.E],
        va[data.bus_fr[e]] - va[data.bus_to[e]] <= data.dvamax[e]
    )

    @objective(model, Min, sum(data.c1[g] * pg[g] for g in 1:data.G))
    return model, (d=d, va=va, pg=pg, pf=pf)
end

function sample_load_scenarios(data::DcopfLoadCase, n::Integer; rng=MersenneTwister(7), rel_width=0.20)
    samples = zeros(Float64, data.L, n)
    total_cap = sum(data.pgmax)
    for j in 1:n
        scale = 1 .+ rel_width .* (2 .* rand(rng, data.L) .- 1)
        samples[:, j] .= data.pd_ref .* scale
        total = sum(@view samples[:, j])
        if total > 0.92 * total_cap
            samples[:, j] .*= (0.92 * total_cap) / total
        end
    end
    return samples
end

function run_dcopf_parametric_load_example(; n_train=48, n_test=16, seed=7)
    data = dcopf_case5()
    model, vars = build_parametric_dcopf_model(data)
    rng = MersenneTwister(seed)

    train = sample_load_scenarios(data, n_train; rng)
    built = build_atlas(model, train; freeze_result=true, atlas_policy=AtlasPolicy(max_certificates=512))
    frozen = built.atlas

    train_ids = query_batch(frozen, train)
    train_primal = primal_batch(frozen, train_ids, train)

    va_rng = 1:data.N
    pg_rng = (data.N + 1):(data.N + data.G)
    pf_rng = (data.N + data.G + 1):(data.N + data.G + data.E)
    max_pg_error = 0.0
    max_pf_error = 0.0
    max_obj_error = 0.0

    for j in axes(train, 2)
        AffineRegions._set_param_vector!(vars.d, @view train[:, j])
        optimize!(model)
        max_pg_error = max(max_pg_error, maximum(abs.(value.(vars.pg) .- train_primal[pg_rng, j])))
        max_pf_error = max(max_pf_error, maximum(abs.(value.(vars.pf) .- train_primal[pf_rng, j])))
        max_obj_error = max(max_obj_error, abs(objective_value(model) - sum(data.c1 .* train_primal[pg_rng, j])))
    end

    test = sample_load_scenarios(data, n_test; rng)
    test_ids = n_test == 0 ? Int[] : query_batch(frozen, test)
    test_coverage = n_test == 0 ? 1.0 : count(!iszero, test_ids) / n_test

    return (
        data=data,
        atlas=frozen,
        build_report=built.report,
        train_coverage=count(!iszero, train_ids) / n_train,
        test_coverage=test_coverage,
        atlas_size=length(frozen.certificates),
        train_metrics=atlas_metrics(frozen),
        max_pg_error=max_pg_error,
        max_pf_error=max_pf_error,
        max_obj_error=max_obj_error,
        variable_ranges=(va=va_rng, pg=pg_rng, pf=pf_rng),
    )
end

function main()
    report = run_dcopf_parametric_load_example()
    println("Parametric DCOPF example")
    println("  atlas_size      = ", report.atlas_size)
    println("  train_coverage  = ", report.train_coverage)
    println("  test_coverage   = ", report.test_coverage)
    println("  max_pg_error    = ", report.max_pg_error)
    println("  max_pf_error    = ", report.max_pf_error)
    println("  max_obj_error   = ", report.max_obj_error)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
