abstract type AffineRegionsError <: Exception end

struct UnsupportedModelError <: AffineRegionsError
    msg::String
end

struct InvalidModelStateError <: AffineRegionsError
    msg::String
end

struct NativeBasisError <: AffineRegionsError
    msg::String
end

struct QueryCoverageError <: AffineRegionsError
    msg::String
end

Base.showerror(io::IO, err::AffineRegionsError) = print(io, err.msg)

struct NumericalPolicy
    zero_tol::Float64
    basis_tol::Float64
    region_tol::Float64
    reduction_tol::Float64
    dedup_tol::Float64
    exact_reduction::Bool
    exact_reduction_param_limit::Int
    exact_reduction_row_limit::Int
end

default_policy() = NumericalPolicy(1e-10, 1e-8, 1e-8, 1e-9, 1e-8, false, 8, 256)

struct AtlasPolicy
    max_certificates::Int
    max_shared_rows::Int
    learning_enabled::Bool
    eviction::Symbol
end

function AtlasPolicy(; max_certificates=typemax(Int), max_shared_rows=typemax(Int), learning_enabled=true, eviction::Symbol=:lru)
    eviction in (:none, :fifo, :lru) ||
        throw(ArgumentError("Unsupported atlas eviction policy $eviction. Supported policies are :none, :fifo, and :lru."))
    max_certificates >= 0 || throw(ArgumentError("Atlas max_certificates must be nonnegative."))
    max_shared_rows >= 0 || throw(ArgumentError("Atlas max_shared_rows must be nonnegative."))
    return AtlasPolicy(max_certificates, max_shared_rows, learning_enabled, eviction)
end

default_atlas_policy() = AtlasPolicy()
