@testitem "EpiProblem assembles and simulates a composed model" begin
    using EpiAwarePrototype, Distributions, Random
    Random.seed!(71)
    data = EpiData([0.2, 0.3, 0.5], exp)
    problem = EpiProblem(
        epi_model = DirectInfections(; data = data, initialisation_prior = Normal()),
        latent_model = RandomWalk(),
        observation_model = PoissonError(),
        tspan = (1, 20))
    m = as_turing_model(problem, (; y_t = missing))
    sim = m()
    @test length(sim.generated_y_t) == 20
    @test length(sim.Z_t) == 20
end

@testitem "apply_method runs a NUTSampler over an EpiProblem" tags=[:sample] begin
    using EpiAwarePrototype, Distributions, Random
    Random.seed!(72)
    data = EpiData([0.2, 0.3, 0.5], exp)
    problem = EpiProblem(
        epi_model = DirectInfections(; data = data, initialisation_prior = Normal()),
        latent_model = RandomWalk(),
        observation_model = PoissonError(),
        tspan = (1, 20))
    ydata = as_turing_model(problem, (; y_t = missing))().generated_y_t
    res = apply_method(problem, NUTSampler(; ndraws = 40, nchains = 1), (; y_t = ydata))
    @test res !== nothing
end

@testitem "EpiMethod threads a Pathfinder pre-step into NUTS" tags=[:sample] begin
    using EpiAwarePrototype, Distributions, Random
    Random.seed!(73)
    data = EpiData([0.2, 0.3, 0.5], exp)
    problem = EpiProblem(
        epi_model = DirectInfections(; data = data, initialisation_prior = Normal()),
        latent_model = RandomWalk(),
        observation_model = PoissonError(),
        tspan = (1, 20))
    ydata = as_turing_model(problem, (; y_t = missing))().generated_y_t
    method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(; ndraws = 10, nruns = 2)],
        sampler = NUTSampler(; ndraws = 40, nchains = 1))
    res = apply_method(problem, method, (; y_t = ydata))
    @test res !== nothing
end

@testitem "spread_draws produces tidy draw/chain/iteration columns" tags=[:sample] begin
    using EpiAwarePrototype, Distributions, Turing, MCMCChains, Random
    Random.seed!(74)
    @model f() = (x ~ Normal())
    chn = MCMCChains.Chains(sample(f(), NUTS(), 30; progress = false))
    df = spread_draws(chn)
    @test all(c -> c in names(df), ["draw", "chain", "iteration"])
    @test size(df, 1) == 30
end

@testitem "generated_observables wraps model, data, and solution" begin
    using EpiAwarePrototype, Distributions, Random
    Random.seed!(75)
    data = EpiData([0.2, 0.3, 0.5], exp)
    m = as_turing_model(
        EpiAwareModel(RandomWalk(),
            DirectInfections(; data = data, initialisation_prior = Normal()),
            PoissonError()), missing, 10)
    obs = generated_observables(m, (; y_t = missing), rand(m))
    @test obs isa EpiAwareObservables
    @test obs.model === m
end
