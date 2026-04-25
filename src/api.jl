function certificate(model::JuMP.Model; policy=default_policy(), tol=policy.basis_tol, exact_reduction=policy.exact_reduction)
    check_model(model)
    lp = extract_lp(model)
    return _build_certificate(lp, model; policy, tol, exact_reduction)
end

function matrix_affine_region(model::JuMP.Model; policy=default_policy(), tol=policy.basis_tol, exact_reduction=policy.exact_reduction)
    return certificate(model; policy, tol, exact_reduction)
end

function affine_region(model::JuMP.Model; policy=default_policy(), tol=policy.basis_tol, exact_reduction=policy.exact_reduction)
    cert = certificate(model; policy, tol, exact_reduction)
    primal_law, dual_law = _laws_to_dicts(cert.P, cert.p0, cert.D, cert.d0, model)
    constraints = _matrix_to_constraints(cert.C, cert.c, extract_lp(model).parameters; tol)
    return AffineRegion(constraints, primal_law, dual_law)
end

function contains_point(cert::BasisCertificate, p::AbstractVector; policy=default_policy(), tol=policy.region_tol)
    vals = cert.C * p
    vals .+= cert.c
    return all(v -> v >= -tol, vals)
end

function contains_batch(cert::BasisCertificate, Pbatch::AbstractMatrix; policy=default_policy(), tol=policy.region_tol)
    vals = cert.C * Pbatch
    vals .+= cert.c
    return vec(all(vals .>= -tol, dims=1))
end

function primal_solution(cert::BasisCertificate, p::AbstractVector)
    x = cert.P * p
    x .+= cert.p0
    return x
end

function primal_batch(cert::BasisCertificate, Pbatch::AbstractMatrix)
    X = cert.P * Pbatch
    X .+= cert.p0
    return X
end

function dual_solution(cert::BasisCertificate, p::AbstractVector)
    λ = cert.D * p
    λ .+= cert.d0
    return λ
end

function dual_batch(cert::BasisCertificate, Pbatch::AbstractMatrix)
    Λ = cert.D * Pbatch
    Λ .+= cert.d0
    return Λ
end
