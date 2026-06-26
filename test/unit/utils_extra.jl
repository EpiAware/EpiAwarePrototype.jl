@testitem "censored_pmf and censored_cdf discretise a distribution" begin
    using EpiAwarePrototype, Distributions
    pmf = censored_pmf(Gamma(2.0, 1.0))
    @test isapprox(sum(pmf), 1.0)
    @test all(>=(0), pmf)

    cdf_ = censored_cdf(Exponential(1.0); D = 10)
    @test cdf_[1] == 0.0
    @test issorted(cdf_)
    @test EpiAwarePrototype.∫F(Exponential(1.0), 2.0, 1.0) > 0
end

@testitem "expected_Rt inverts the renewal relationship" begin
    using EpiAwarePrototype
    data = EpiData([0.2, 0.3, 0.5], exp)
    rt = expected_Rt(data, [100.0, 200, 300, 400, 500])
    @test length(rt) == 2
    @test all(>(0), rt)
end

@testitem "DirectSample draws from the prior" begin
    using EpiAwarePrototype, Distributions, Turing, Random
    Random.seed!(81)
    @model g() = (x ~ Normal())
    # apply_method wraps the solution in EpiAwareObservables; `.samples` is the
    # raw inference result.
    chain = apply_method(g(), DirectSample(; n_samples = 10))
    @test chain isa EpiAwareObservables
    @test chain.samples !== nothing
    single = apply_method(g(), DirectSample())
    @test haskey(single.samples, @varname(x))
end

@testitem "get_param_array reshapes a Chains into (draws, chains)" begin
    using EpiAwarePrototype, Distributions, Turing, MCMCChains, Random
    Random.seed!(82)
    @model g() = (x ~ Normal())
    chn = MCMCChains.Chains(sample(g(), Prior(), MCMCSerial(), 3, 2; progress = false))
    A = get_param_array(chn)
    @test size(A) == (3, 2)
end
