# Adaptive Traffic Routing with Attention Networks

Aicha Rahmani1*, Noureddine Abghour1 and Zouhair Chiba1

*Correspondence: a.rahmani@example.org

1 Laboratory of Intelligent Systems, Faculty of Sciences, Casablanca, Morocco

Published online: 14 March 2021

## Abstract

We introduce ARTNet, a routing model for coordinating signalized intersections in
dense urban networks. ARTNet couples an attention mechanism with multi-agent
reinforcement learning, allowing each intersection to condition its phase decisions
on the observed state of its immediate neighbours rather than on a central plan. We
evaluate on two public benchmarks and report reductions in average vehicle delay of
23.4% against a fixed-time controller and 9.1% against MaxPressure.

## Introduction

Signal control at urban intersections is usually handled either by fixed-time plans
computed offline or by actuated controllers that respond to local detector counts.
Neither adapts well to demand that shifts across the day, and neither coordinates
across intersections without a central planner.

Reinforcement learning offers an alternative, but naive multi-agent formulations
scale poorly: each agent's environment is non-stationary because its neighbours are
learning at the same time. Our contribution is to give each agent an explicit
attention mechanism over its neighbours' observations, which stabilises training
without requiring a central critic.

## Related Work

Max pressure control derives phase decisions from queue differentials and comes with
throughput guarantees under mild assumptions. Learned approaches have since shown
gains on simulated networks, generally by treating each intersection as an
independent agent and accepting the resulting non-stationarity. Work on graph-based
coordination shares our motivation but typically aggregates neighbour state with a
fixed weighting rather than a learned one.

## Method

### Model Architecture

ARTNet trains a multi-agent deep Q-network with a shared replay buffer. Each agent
observes queue length and current phase duration at its own intersection, and
attends over the same observations from intersections within two hops. Attention
weights are produced by a single-layer scaled dot-product head, so the number of
parameters does not grow with network size.

### Training Details

We optimize with Adam at a learning rate of 0.0003 for 500 episodes, using a replay
buffer of one million transitions and a batch size of 64. Exploration follows an
epsilon-greedy schedule annealed from 1.0 to 0.05 over the first 200 episodes.
Target network weights are updated every 1000 gradient steps.

## Experiments

### Datasets

We evaluate on the Hangzhou and Jinan benchmarks, containing 16 and 12 signalized
intersections respectively, with demand traces recorded over a full week. Both are
distributed with the CityFlow simulator.

### Baselines

We compare against a fixed-time controller tuned by Webster's method, MaxPressure,
and a single-agent DQN that observes the whole network state.

### Evaluation Metrics

We report average vehicle delay in seconds, network throughput in vehicles per hour,
and average queue length at the end of each phase.

## Results

ARTNet reduced average vehicle delay by 23.4% against the fixed-time baseline and by
9.1% against MaxPressure on Hangzhou. Throughput improved by 11.8%. On Jinan, which
has a more irregular topology, the margin against MaxPressure narrowed to 4.2%,
which we attribute to the shorter average distance between intersections.

## Discussion

The attention weights are interpretable: on Hangzhou they concentrate on upstream
intersections during peak inbound flow and redistribute during the evening reversal.
This suggests the model recovers the structure a traffic engineer would impose by
hand, without being given it.

## Conclusion and Future Work

Attention-based routing coordinates intersections without central control and
without a central critic. The obvious limitation is that we evaluate in simulation
only; deployment would need to handle detector noise and dropped observations, which
we have not modelled.

## References

[1] Varaiya, P. Max pressure control of a network of signalized intersections. 2013.
[2] Wei, H. et al. PressLight: learning max pressure control for signalized
    intersections. 2019.
[3] Chen, C. et al. Toward a thousand lights: decentralized deep reinforcement
    learning for large-scale traffic signal control. 2020.
[4] Zheng, G. et al. Learning phase competition for traffic signal control. 2019.
[5] Oroojlooy, A. et al. AttendLight: universal attention-based reinforcement
    learning model for traffic signal control. 2020.
[6] Zhang, H. et al. CityFlow: a multi-agent reinforcement learning environment for
    large scale city traffic scenario. 2019.
[7] Webster, F. V. Traffic signal settings. 1958.
