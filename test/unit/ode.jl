@testitem "SIRParams and SEIRParams sample (u0, p)" begin
    using EpiAwarePrototype, OrdinaryDiffEq, Distributions, Random
    Random.seed!(61)
    sir = SIRParams(tspan = (0.0, 30.0),
        infectiousness = LogNormal(log(0.3), 0.05),
        recovery_rate = LogNormal(log(0.1), 0.05),
        initial_prop_infected = Beta(1, 99))
    u0, p = as_turing_model(sir, nothing)()
    @test length(u0) == 3
    @test length(p) == 2
    @test isapprox(sum(u0), 1.0; atol = 1e-8)

    seir = SEIRParams(tspan = (0.0, 30.0),
        infectiousness = LogNormal(log(0.3), 0.05),
        incubation_rate = LogNormal(log(0.1), 0.05),
        recovery_rate = LogNormal(log(0.1), 0.05),
        initial_prop_infected = Beta(1, 99))
    u0s, ps = as_turing_model(seir, nothing)()
    @test length(u0s) == 4
    @test length(ps) == 3
    @test isapprox(sum(u0s), 1.0; atol = 1e-8)
end

@testitem "ODEProcess solves the SIR model into an infection series" begin
    using EpiAwarePrototype, OrdinaryDiffEq, Distributions, LogExpFunctions, Random
    Random.seed!(62)
    sir = SIRParams(tspan = (0.0, 100.0),
        infectiousness = LogNormal(log(0.3), 0.05),
        recovery_rate = LogNormal(log(0.1), 0.05),
        initial_prop_infected = Beta(1, 99))
    N = 1000.0
    proc = ODEProcess(params = sir,
        sol2infs = sol -> softplus.(N .* sol[2, :]),
        solver_options = Dict(:saveat => 1.0))
    I_t = as_turing_model(proc, nothing)()
    @test length(I_t) == 101
    @test all(>=(0), I_t)
end

@testitem "ODEProcess samples its parameters from the prior" begin
    using EpiAwarePrototype, OrdinaryDiffEq, Distributions, LogExpFunctions, Random
    Random.seed!(63)
    sir = SIRParams(tspan = (0.0, 50.0),
        infectiousness = LogNormal(log(0.3), 0.05),
        recovery_rate = LogNormal(log(0.1), 0.05),
        initial_prop_infected = Beta(1, 99))
    proc = ODEProcess(params = sir, sol2infs = sol -> 1000.0 .* sol[2, :],
        solver_options = Dict(:saveat => 1.0))
    draw = rand(as_turing_model(proc, nothing))
    names = string.(collect(keys(draw)))
    @test "β" in names
    @test "γ" in names
end
