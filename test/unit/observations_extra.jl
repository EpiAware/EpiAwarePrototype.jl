@testitem "Ascertainment scales expected observations by a latent model" begin
    using EpiAwarePrototype, Distributions, Random
    Random.seed!(51)
    Y = fill(100.0, 14)
    asc = Ascertainment(model = NegativeBinomialError(), latent_model = FixedIntercept(0.1))
    sim = as_turing_model(asc, missing, Y)()
    @test length(sim) == length(Y)
    @test all(>=(0), sim)

    adw = ascertainment_dayofweek(PoissonError())
    @test length(as_turing_model(adw, missing, Y)()) == length(Y)
end

@testitem "Aggregate sums expected observations over windows" begin
    using EpiAwarePrototype, Random
    Random.seed!(52)
    agg = Aggregate(PoissonError(), [0, 0, 0, 0, 7, 0, 0])
    out = as_turing_model(agg, missing, fill(1.0, 28))()
    @test length(out) == 28
    # Only the present (weekly) positions are non-zero.
    @test count(!=(0), out) == 4
end

@testitem "PrefixObservationModel prefixes observation parameters" begin
    using EpiAwarePrototype, Random
    Random.seed!(53)
    pom = PrefixObservationModel(model = NegativeBinomialError(), prefix = "Test")
    names = string.(collect(keys(rand(as_turing_model(pom, missing, fill(10.0, 5))))))
    @test any(startswith("Test."), names)
end

@testitem "RecordExpectedObs and TransformObservationModel wrap an error model" begin
    using EpiAwarePrototype, Random
    Random.seed!(54)
    Y = fill(10.0, 30)
    reo = RecordExpectedObs(NegativeBinomialError())
    @test length(as_turing_model(reo, missing, Y)()) == length(Y)

    tom = TransformObservationModel(NegativeBinomialError())
    @test length(as_turing_model(tom, missing, Y)()) == length(Y)
end

@testitem "StackObservationModels prefixes and stacks several models" begin
    using EpiAwarePrototype, Distributions, Random
    Random.seed!(55)
    stk = StackObservationModels((cases = PoissonError(),
        deaths = NegativeBinomialError()))
    yt = (cases = missing, deaths = missing)
    sm = as_turing_model(stk, yt, fill(10.0, 10))
    names = string.(collect(keys(rand(sm))))
    @test any(startswith("cases."), names)
    @test any(startswith("deaths."), names)
    out = sm()
    @test length(out) == 2
    @test length(out[1]) == 10
end
