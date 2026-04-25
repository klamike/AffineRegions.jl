function build_atlas!(
    atlas::RegionAtlas,
    model::JuMP.Model,
    sample_points::AbstractMatrix;
    policy=atlas.policy,
    exact_reduction=policy.exact_reduction,
    require_feasible=true,
)
    lp = extract_lp(model)
    size(sample_points, 1) == length(lp.parameters) ||
        throw(DimensionMismatch("Sample matrix has $(size(sample_points, 1)) parameter rows; expected $(length(lp.parameters))."))

    added0 = _metric_get(atlas.metrics, :certificates_added)
    feasible0 = _metric_get(atlas.metrics, :builder_feasible)
    skipped0 = _metric_get(atlas.metrics, :builder_skipped)
    points0 = _metric_get(atlas.metrics, :builder_points)

    for j in axes(sample_points, 2)
        _metric_inc!(atlas.metrics, :builder_points)
        p = @view sample_points[:, j]
        _set_param_vector!(lp.parameters, p)
        JuMP.optimize!(model)
        if JuMP.is_solved_and_feasible(model)
            _metric_inc!(atlas.metrics, :builder_feasible)
            cert = certificate(model; policy=policy, exact_reduction=exact_reduction)
            add_certificate!(atlas, cert)
        elseif require_feasible
            throw(InvalidModelStateError("Offline atlas build encountered a non-feasible sample at column $j with termination status $(JuMP.termination_status(model))."))
        else
            _metric_inc!(atlas.metrics, :builder_skipped)
        end
    end

    attempted = _metric_get(atlas.metrics, :builder_points) - points0
    feasible = _metric_get(atlas.metrics, :builder_feasible) - feasible0
    skipped = _metric_get(atlas.metrics, :builder_skipped) - skipped0
    added = _metric_get(atlas.metrics, :certificates_added) - added0
    return AtlasBuildReport(
        attempted,
        feasible,
        skipped,
        added,
        max(feasible - added, 0),
        length(atlas.certificates),
        false,
    )
end

function build_atlas!(
    atlas::RegionAtlas,
    model::JuMP.Model,
    sample_points::AbstractVector{<:AbstractVector};
    kwargs...,
)
    isempty(sample_points) && return AtlasBuildReport(0, 0, 0, 0, 0, length(atlas.certificates), false)
    return build_atlas!(atlas, model, reduce(hcat, sample_points); kwargs...)
end

function build_atlas(
    model::JuMP.Model,
    sample_points;
    policy=default_policy(),
    atlas_policy=default_atlas_policy(),
    freeze_result=false,
    kwargs...,
)
    atlas = RegionAtlas(; policy=policy, atlas_policy=atlas_policy)
    report = build_atlas!(atlas, model, sample_points; policy=policy, kwargs...)
    if freeze_result
        return (atlas=freeze(atlas), report=AtlasBuildReport(report.attempted_points, report.feasible_points, report.skipped_points, report.added_certificates, report.deduplicated_points, report.final_certificates, true))
    end
    return (atlas=atlas, report=report)
end
