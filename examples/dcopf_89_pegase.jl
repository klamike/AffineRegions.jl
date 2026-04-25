using AffineRegions
using DiffOpt
using HiGHS
using JuMP
using PGLib
using PowerModels
using Random

struct DcopfLoadCase
    case_name::String
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
    load_bus::Vector{Int}
    load_group::Vector{Int}
    n_groups::Int
    bus_arcs_fr::Vector{Vector{Int}}
    bus_arcs_to::Vector{Vector{Int}}
    bus_gens::Vector{Vector{Int}}
    bus_loads::Vector{Vector{Int}}
    gen_status::Vector{Bool}
    branch_status::Vector{Bool}
end

function load_pegase_case89(; n_groups=6)
    case_file = joinpath(PGLib.PGLib_opf, first(PGLib.find_pglib_case("89_pegase")))
    network = PowerModels.make_basic_network(PowerModels.parse_file(case_file))

    N = length(network["bus"])
    E = length(network["branch"])
    G = length(network["gen"])
    L = length(network["load"])

    ref_bus = only(i for i in 1:N if network["bus"]["$i"]["bus_type"] == 3)

    gs = zeros(Float64, N)
    for (_, shunt) in network["shunt"]
        shunt["status"] == 1 || continue
        gs[shunt["shunt_bus"]] += shunt["gs"]
    end

    bus_fr = zeros(Int, E)
    bus_to = zeros(Int, E)
    b = zeros(Float64, E)
    smax = zeros(Float64, E)
    dvamin = zeros(Float64, E)
    dvamax = zeros(Float64, E)
    branch_status = zeros(Bool, E)
    bus_arcs_fr = [Int[] for _ in 1:N]
    bus_arcs_to = [Int[] for _ in 1:N]
    for e in 1:E
        branch = network["branch"]["$e"]
        i = branch["f_bus"]
        j = branch["t_bus"]
        z = branch["br_r"] + im * branch["br_x"]
        y = inv(z)
        isfinite(y) || error("Branch $e has zero impedance.")

        bus_fr[e] = i
        bus_to[e] = j
        b[e] = imag(y)
        smax[e] = branch["rate_a"]
        dvamin[e] = branch["angmin"]
        dvamax[e] = branch["angmax"]
        branch_status[e] = branch["br_status"] == 1
        push!(bus_arcs_fr[i], e)
        push!(bus_arcs_to[j], e)
    end

    gen_bus = zeros(Int, G)
    pgmin = zeros(Float64, G)
    pgmax = zeros(Float64, G)
    c1 = zeros(Float64, G)
    gen_status = zeros(Bool, G)
    bus_gens = [Int[] for _ in 1:N]
    for g in 1:G
        gen = network["gen"]["$g"]
        i = gen["gen_bus"]
        gen_bus[g] = i
        pgmin[g] = gen["pmin"]
        pgmax[g] = gen["pmax"]
        c1[g] = gen["cost"][2]
        gen_status[g] = gen["gen_status"] == 1
        push!(bus_gens[i], g)
    end

    pd_ref = zeros(Float64, L)
    load_bus = zeros(Int, L)
    bus_loads = [Int[] for _ in 1:N]
    for l in 1:L
        load = network["load"]["$l"]
        i = load["load_bus"]
        load_bus[l] = i
        pd_ref[l] = (load["status"] == 1) * load["pd"]
        push!(bus_loads[i], l)
    end

    load_group = zeros(Int, L)
    bus_order = sortperm(load_bus)
    edges = round.(Int, range(0, L; length=n_groups + 1))
    for group in 1:n_groups
        lo = edges[group] + 1
        hi = edges[group + 1]
        for idx in lo:hi
            idx > L && continue
            load_group[bus_order[idx]] = group
        end
    end
    any(==(0), load_group) && error("Load grouping failed.")

    return DcopfLoadCase(
        basename(case_file),
        N,
        E,
        G,
        L,
        ref_bus,
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
        load_bus,
        load_group,
        n_groups,
        bus_arcs_fr,
        bus_arcs_to,
        bus_gens,
        bus_loads,
        gen_status,
        branch_status,
    )
end

function build_parametric_dcopf_model(data::DcopfLoadCase; optimizer=HiGHS.Optimizer)
    model = DiffOpt.basis_diff_model(optimizer)
    JuMP.set_silent(model)

    @variable(model, α[g in 1:data.n_groups] in Parameter(1.0))
    @variable(model, va[1:data.N])
    @variable(model, pg[1:data.G])
    @variable(model, pf[1:data.E])

    set_lower_bound.(pg, data.gen_status .* data.pgmin)
    set_upper_bound.(pg, data.gen_status .* data.pgmax)
    set_lower_bound.(pf, data.branch_status .* .-data.smax)
    set_upper_bound.(pf, data.branch_status .* data.smax)

    @expression(model, pd[l in 1:data.L], data.pd_ref[l] * α[data.load_group[l]])
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
    return model, (alpha=α, va=va, pg=pg, pf=pf)
end

function sample_group_load_scenarios(data::DcopfLoadCase, n::Integer; rng=MersenneTwister(17), rel_width=0.30)
    return 1 .+ rel_width .* (2 .* rand(rng, data.n_groups, n) .- 1)
end

function sample_feasible_group_load_scenarios(
    model::JuMP.Model,
    alpha,
    data::DcopfLoadCase,
    n::Integer;
    rng=MersenneTwister(17),
    rel_width=0.20,
    max_tries=max(10n, 1_000),
)
    samples = zeros(Float64, data.n_groups, n)
    accepted = 0
    attempted = 0
    while accepted < n && attempted < max_tries
        attempted += 1
        candidate = 1 .+ rel_width .* (2 .* rand(rng, data.n_groups) .- 1)
        AffineRegions._set_param_vector!(alpha, candidate)
        optimize!(model)
        if is_solved_and_feasible(model; dual=false)
            accepted += 1
            samples[:, accepted] .= candidate
        end
    end
    accepted == n || error("Failed to find $n feasible load scenarios in $attempted attempts.")
    return samples, attempted
end

function run_dcopf_89_pegase_example(;
    n_groups=6,
    n_train=500,
    n_test=2000,
    rel_width=0.20,
    seed=17,
    verify_test::Bool=false,
)
    data = load_pegase_case89(; n_groups)
    model, vars = build_parametric_dcopf_model(data)
    rng = MersenneTwister(seed)

    train, train_attempts = sample_feasible_group_load_scenarios(model, vars.alpha, data, n_train; rng, rel_width)
    built = build_atlas(
        model,
        train;
        freeze_result=true,
        atlas_policy=AtlasPolicy(max_certificates=4096, eviction=:lru),
    )
    frozen = built.atlas

    train_ids = query_batch(frozen, train)
    train_coverage = count(!iszero, train_ids) / n_train

    if n_test > 0
        test, test_attempts = sample_feasible_group_load_scenarios(model, vars.alpha, data, n_test; rng, rel_width)
        test_ids = query_batch(frozen, test)
        test_coverage = count(!iszero, test_ids) / n_test
    else
        test = zeros(Float64, data.n_groups, 0)
        test_attempts = 0
        test_ids = Int[]
        test_coverage = 1.0
    end

    covered = findall(!iszero, test_ids)
    max_pg_error = verify_test ? 0.0 : missing
    max_pf_error = verify_test ? 0.0 : missing
    max_obj_error = verify_test ? 0.0 : missing
    primal = (verify_test && !isempty(covered)) ? primal_batch(frozen, test_ids[covered], test[:, covered]) : nothing

    pg_rng = (data.N + 1):(data.N + data.G)
    pf_rng = (data.N + data.G + 1):(data.N + data.G + data.E)
    if verify_test
        @assert primal !== nothing
        for (col, j) in enumerate(covered)
            AffineRegions._set_param_vector!(vars.alpha, @view test[:, j])
            optimize!(model)
            max_pg_error = max(max_pg_error, maximum(abs.(value.(vars.pg) .- primal[pg_rng, col])))
            max_pf_error = max(max_pf_error, maximum(abs.(value.(vars.pf) .- primal[pf_rng, col])))
            max_obj_error = max(max_obj_error, abs(objective_value(model) - sum(data.c1 .* primal[pg_rng, col])))
        end
    end

    return (
        data=data,
        atlas=frozen,
        build_report=built.report,
        atlas_size=length(frozen.certificates),
        train_coverage=train_coverage,
        test_coverage=test_coverage,
        train_attempts=train_attempts,
        test_attempts=test_attempts,
        covered_test_points=length(covered),
        uncovered_test_points=n_test - length(covered),
        verify_test=verify_test,
        max_pg_error=max_pg_error,
        max_pf_error=max_pf_error,
        max_obj_error=max_obj_error,
        metrics=atlas_metrics(frozen),
    )
end

function main()
    report = run_dcopf_89_pegase_example()
    println("Parametric DCOPF example (89_pegase)")
    println("  atlas_size            = ", report.atlas_size)
    println("  train_coverage        = ", report.train_coverage)
    println("  test_coverage         = ", report.test_coverage)
    println("  train_attempts        = ", report.train_attempts)
    println("  test_attempts         = ", report.test_attempts)
    println("  covered_test_points   = ", report.covered_test_points)
    println("  uncovered_test_points = ", report.uncovered_test_points)
    println("  verify_test           = ", report.verify_test)
    if report.verify_test
        println("  max_pg_error          = ", report.max_pg_error)
        println("  max_pf_error          = ", report.max_pf_error)
        println("  max_obj_error         = ", report.max_obj_error)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
