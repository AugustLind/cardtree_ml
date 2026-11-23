# Learning card placement in a binary search tree

The curiosity began with a **kortstokk** (a deck of cards) in the mountains of
Nepal. A deck was helping teach a friend about binary search trees; the next
question was whether a machine could learn to place cards correctly from
feedback.

## Experiment

The agent works with one card per rank, Ace low. Each round seeds a fresh tree
with 4–10 ranks, then asks the agent to place each remaining rank. A correct
card enters the tree and earns `+0.2`; a wrong card is discarded and receives
the negative of its penalty. One placement attempt is one training iteration.
Training is considered complete only after three perfect rounds in a row; any
imperfect round resets the streak. The streak and completion iteration persist
with the learned policy.

The learner is a small tabular, episodic Q-learning-style agent. A state is
`(incoming rank, current node rank)` and its actions are `L` or `R`. It chooses
with epsilon-greedy exploration (`ε = 0.25`, decaying by `0.995` per attempt
to a floor of `0.05`). After a card reaches a slot, its terminal reward is
propagated backward through the comparisons:

```text
G_t = γ^(T - t - 1) R
Q(s_t, a_t) ← Q(s_t, a_t) + α (G_t - Q(s_t, a_t))
```

Here `α = 0.2`, `γ = 0.9`, and `R` is the final placement reward. This is a
compact episodic update rather than a general-purpose neural model; its table
learns the preferred direction for observed rank pairs.

## Penalty math

Let `a` be the attempted path and `c` the correct path, each a sequence of
`L`/`R` directions from the root. Let `d` be the length of their shared prefix
before the first wrong branch. Let `D = |a| + |c| - 2d`, the edge distance
between the two slots. The benchmark compares these costs on wrong placements:

| Cost rule | Penalty |
| --- | ---: |
| Current: inverse divergence depth | `1 / (d + 1)` |
| Constant | `1` |
| Root-focused exponential | `2^(-d)` |
| Root-focused linear | `max(0.1, 1 - 0.1d)` |
| Root-focused inverse square | `1 / (d + 1)^2` |
| Deeper mistakes cost more | `d + 1` |
| Tree edge distance | `D` |
| Squared tree distance | `D^2` |
| Distance with root weighting | `D / (d + 1)` |

The current rule makes an early wrong turn more costly: errors at the root cost
`1`, one level deeper cost `1/2`, and two levels deeper cost `1/3`.

## Initial benchmark finding

The app benchmark trains five fresh agents per rule, using a fixed matched
random-seed schedule and stopping each run after its first three-round perfect
streak (or after 5,000 rounds). In one run every rule completed in all five
trials. Mean placement iterations to the three-round streak were:

| Cost rule | Mean iterations |
| --- | ---: |
| Root-focused exponential | 506.8 |
| Current: inverse divergence depth | 509.4 |
| Root-focused inverse square | 569.4 |
| Root-focused linear | 603.2 |
| Constant | 638.6 |
| Distance with root weighting | 656.0 |
| Tree edge distance | 739.6 |
| Squared tree distance | 757.2 |
| Deeper mistakes cost more | 6598.2 |

This small, deterministic benchmark suggests that root-weighted costs reached
the three-round streak sooner than costs that increase with mistake depth. It
is an early finding, not a general result: the deck is small, there are only
five trials per rule, and outcomes depend on the learner and seed schedule.
