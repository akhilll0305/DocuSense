# Queue-Aware Signal Control Without Learned Policies

Priya Raghavan1, Tomas Lindqvist2 and Wei Chen1

1 Department of Transport Engineering, Northgate University
2 Institute for Urban Systems, Lund

Published online: 2 September 2022

## Abstract

We revisit classical queue-based signal control and ask how much of the gain
attributed to reinforcement learning survives a well-tuned analytical baseline. Using
the same two benchmarks reported in recent learned-control papers, we find that a
queue-aware controller with per-approach saturation estimates closes most of the gap,
reducing average vehicle delay by 19.7% against fixed-time signals without any
training phase.

## Introduction

Recent work reports large improvements from learned signal controllers, usually
measured against a fixed-time plan. Fixed-time plans are a weak baseline: they are
computed once and never adapt. We argue that the interesting comparison is against an
analytical controller that uses the same observations the learned agent gets.

## Related Work

Learned controllers for signal timing generally frame the problem as a Markov
decision process over queue lengths and phase indicators. Reported gains range from
10% to 25% delay reduction. Analytical approaches, by contrast, have been treated as
a fixed reference point rather than as a design target.

## Method

### Problem Formulation

We model each intersection as a set of approaches with time-varying arrival rates and
per-approach saturation flow. The controller chooses the next phase and its duration
to minimise expected cumulative delay over a rolling horizon of 90 seconds.

### Saturation Estimation

Saturation flow is estimated online from detector occupancy over a trailing 15-minute
window, rather than taken from a design manual. This is the single change that
accounts for most of our improvement over textbook queue control.

## Experiments

### Datasets

We use the Hangzhou and Jinan benchmarks distributed with CityFlow, matching the
setup used by recent learned-control work so the numbers are comparable.

### Baselines

We compare against fixed-time control tuned by Webster's method, textbook
queue-length control with manual saturation constants, and two published learned
controllers whose results we reproduce from released code.

## Results

The queue-aware controller reduced average vehicle delay by 19.7% against fixed-time
signals on Hangzhou, against 23.4% reported for attention-based reinforcement
learning on the same benchmark. On Jinan the two are within 1.5% of each other. The
learned controllers retain an advantage under the most irregular demand traces.

## Discussion

Our reading is not that learned control is unnecessary, but that the margin is
narrower than the fixed-time comparison suggests, and that it comes at the cost of a
training phase, a simulator, and a policy that cannot be inspected. Where the demand
pattern is stable, the analytical controller is the better engineering choice.

## Limitations

We evaluate in simulation, on two networks, both from the same distribution of urban
topologies. We have not tested transferability to networks with unsignalised
interactions or dedicated transit phases.

## Conclusion

Much of the reported gain from learned signal control is recoverable analytically
once saturation flow is estimated online rather than assumed.

## References

[1] Webster, F. V. Traffic signal settings. 1958.
[2] Varaiya, P. Max pressure control of a network of signalized intersections. 2013.
[3] Wei, H. et al. PressLight: learning max pressure control for signalized
    intersections. 2019.
[4] Zhang, H. et al. CityFlow: a multi-agent reinforcement learning environment for
    large scale city traffic scenario. 2019.
[5] Roess, R. P., Prassas, E. S. and McShane, W. R. Traffic Engineering. 2019.
[6] Lin, S. et al. Efficient network-wide model-based predictive control for urban
    traffic networks. 2012.
