# cardtree_ml

The curiosity began with a **kortstokk** (a deck of cards) in the mountains of
Nepal. While using it to teach a friend about binary search trees, we wondered:
could a machine learn to place cards correctly in a BST from feedback?

This project explores that question with a small reinforcement-learning agent.
It practices rank-based card insertion, sees a reward or penalty after each
placement, and displays its tree as playing cards. See
[`docs/cardtree-learning.md`](docs/cardtree-learning.md) for the model, cost
math, and initial benchmark findings.

## Run the app

Use Python 3.10 or later with Tkinter available:

```sh
python3 -m cardtree_ml
```

The robot practices BST insertion with one card for each rank (Ace is low).
Each round starts with 4–10 randomly selected cards already placed in a fresh
tree; the robot attempts each remaining card once. The playing-card spades are
visual decoration; rank alone determines ordering. Incorrect cards are shown
with a penalty and discarded. The app counts each placement iteration and
requires three perfect rounds in a row to mark training complete. The learned
Q-values and counters are saved in `~/.cardtree_ml/policy.json` and reused in
later rounds and runs.

The visible robot pauses for about two seconds between placements. After a
round, **Train until 3 perfect rounds** runs shuffled rounds without animation
until every remaining placement is correct in three rounds in a row, then
automatically starts a fresh visible round. It stops after 5,000 rounds if it
has not reached the streak, and leaves the learned policy saved.

After a visible round, **Benchmark punishments** compares nine penalty rules.
It trains five fresh learners per rule with matched random seeds, up to 5,000
rounds per learner, and then shows three-round completion rate plus mean and
median placement iterations alongside each rule's formula. This benchmark
does not modify the saved learner.

## Punishment rule

The starter punishment in `cardtree_ml/punishment.py` is larger when the
attempted path first diverges from the correct path closer to the root. Each
placement is a tuple of `"L"` and `"R"` directions from the root, such as
`("L", "R")`; the root slot is `()`. The function is called only for an
incorrect placement. Its negative is the robot's reward for that attempt. You
can change this simple rule as you experiment with better penalties.
