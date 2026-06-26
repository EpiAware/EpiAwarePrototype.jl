# API reference

## Core architecture

```@docs
as_turing_model
AbstractEpiAwareModel
```

## Latent models

```@docs
IID
HierarchicalNormal
RandomWalk
AR
MA
Intercept
FixedIntercept
Null
DiffLatentModel
```

## Latent modifiers, manipulators, and combinations

```@docs
TransformLatentModel
PrefixLatentModel
RecordExpectedLatent
CombineLatentModels
ConcatLatentModels
BroadcastLatentModel
RepeatEach
RepeatBlock
broadcast_rule
broadcast_n
broadcast_dayofweek
broadcast_weekly
equal_dimensions
arma
arima
```

## Infection models

```@docs
EpiData
DirectInfections
ExpGrowthRate
Renewal
R_to_r
r_to_R
```

## ODE compartmental models

```@docs
SIRParams
SEIRParams
ODEProcess
```

## Observation models

```@docs
PoissonError
NegativeBinomialError
LatentDelay
observation_error
generate_observation_error_priors
```

## Observation modifiers and manipulators

```@docs
Ascertainment
ascertainment_dayofweek
Aggregate
PrefixObservationModel
RecordExpectedObs
TransformObservationModel
StackObservationModels
```

## Composition

```@docs
EpiAwareModel
```

## Utilities and distributions

```@docs
accumulate_scan
get_state
HalfNormal
SafePoisson
SafeNegativeBinomial
NegativeBinomialMeanClust
censored_pmf
condition_model
```
