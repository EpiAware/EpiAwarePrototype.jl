@testitem "ExpGrowthRate maps a growth-rate path to infections" begin
    using EpiAwarePrototype, Distributions, Random
    Random.seed!(41)
    data = EpiData([0.2, 0.3, 0.5], exp)
    egr = ExpGrowthRate(; data = data, initialisation_prior = Normal())
    I_t = as_turing_model(egr, randn(20) * 0.05)()
    @test length(I_t) == 20
    @test all(>(0), I_t)
end

@testitem "Renewal maps an Rt path to infections" begin
    using EpiAwarePrototype, Distributions, Random
    Random.seed!(42)
    data = EpiData([0.2, 0.3, 0.5], exp)
    renewal = Renewal(data; initialisation_prior = Normal())
    I_t = as_turing_model(renewal, randn(20) * 0.05)()
    @test length(I_t) == 20
    @test all(isfinite, I_t)
    @test all(>=(0), I_t)
end

@testitem "growth-rate / reproduction-number conversions round-trip" begin
    using EpiAwarePrototype
    w = [0.2, 0.3, 0.5]
    r = R_to_r(1.5, w)
    @test r_to_R(r, w) ≈ 1.5 rtol=1e-3
    # r and R move in the same direction.
    @test R_to_r(2.0, w) > R_to_r(1.2, w)
end

@testitem "composed Renewal model runs a short NUTS sample" tags=[:sample] begin
    using EpiAwarePrototype, Distributions, Turing, Random
    Random.seed!(43)
    data = EpiData([0.2, 0.3, 0.5], exp)
    model = EpiAwareModel(RandomWalk(),
        Renewal(data; initialisation_prior = Normal()),
        PoissonError())
    y = as_turing_model(model, missing, 20)().generated_y_t
    chn = sample(as_turing_model(model, y, 20), NUTS(), 30; progress = false)
    @test chn !== nothing
end
