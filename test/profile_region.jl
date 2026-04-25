using AffineRegions
using JuMP, HiGHS
using Profile
import Random

function build_and_solve(; n=2000, m=1000, k=10, seed=42)
    Random.seed!(seed)
    model = AffineRegions.Model(HiGHS.Optimizer)

    @variable(model, x[1:n] >= 0)
    params = [@variable(model, set = Parameter(0.0)) for _ in 1:k]

    A = randn(m, n)
    base = randn(m)
    alphas = 0.5 .* randn(m, k)
    for i in 1:m
        @constraint(model, sum(A[i,j] * x[j] for j in 1:n) >= base[i] + sum(alphas[i,q] * params[q] for q in 1:k))
    end

    c = rand(n) .+ 0.1
    @objective(model, Min, sum(c[j] * x[j] for j in 1:n))

    println("Problem: n=$n, m=$m, k=$k")

    # warmup
    optimize!(model)
    for q in 1:k
        set_parameter_value(params[q], 0.1 * q)
    end
    optimize!(model)

    # profile many region formations
    Profile.clear()
    @profile for i in 1:30
        for q in 1:k
            set_parameter_value(params[q], 10.0 * i * q)
        end
        optimize!(model)
    end

    Profile.print(mincount=10, noisefloor=2.0)
end

build_and_solve()
