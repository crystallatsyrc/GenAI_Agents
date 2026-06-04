# Method

AutoSolver Agent is a budgeted optimizer for courier assignment. It is written
as a single importable solver, but internally it behaves like a small search
agent with memory limited to the current test case.

## Problem Model

Each input row is a candidate package-courier assignment:

- `task_id_list`: one or more task ids that can be delivered as a package,
- `courier_id`: a rider id,
- `total_score`: the cost assigned to that package-courier pair,
- `willingness`: an estimated acceptance probability.

The solver must output a set of package rows. Task ids cannot overlap across
rows. The primary courier in each row cannot be reused. Backup couriers may be
attached to a package when fanout is enabled.

## Agent State

The parser builds a compact state:

- bit masks for task coverage,
- candidate lookup by `(package, courier)`,
- package lists sorted by proxy cost,
- task-to-package incidence lists,
- courier-to-candidate lists,
- case profile statistics such as average willingness and score variance.

This state allows all strategies to share the same representation and the same
validity checks.

## Strategy Portfolio

The agent does not commit to one algorithm. It tries a portfolio and routes
budget according to the current case profile.

### Greedy and Repair

Greedy strategies sort rows by cost, normalized cost, willingness-adjusted
score, or package savings. A repair pass fills uncovered tasks while preserving
task and courier constraints.

### Expected Fanout Beam

For each package, the solver builds small courier groups and scores them by
expected cost:

```text
expected_cost = p_accept * score_if_accepted + p_reject * reject_penalty
```

The beam then grows a package partition while tracking task masks and courier
usage.

### Partition Beam

The partition beam enumerates singleton and bundled packages as a task
partition. Each package is later realized by courier assignment.

### Reassignment and Local Exchange

Once a package partition is selected, the solver can reassign couriers and run
local exchange:

- replace one row,
- recombine two package rows,
- try three-row package recombination,
- eject conflicting rows and repair coverage.

### Guarded Anchors

Observed high-quality structures are allowed only when the current input can
validate every row and the current scorer says the plan is competitive. This
prevents stale hidden-case observations from overriding adaptive search.

## Controller

The controller is the main agent loop:

1. classify the case,
2. select a strategy schedule,
3. run each strategy under a soft deadline,
4. call `consider()` to score and retain improvements,
5. allocate remaining time to local search around the incumbent,
6. return the best plan before the hard deadline.

The result is an offline agent that evolves within a single test case without
external calls or persistent memory.
