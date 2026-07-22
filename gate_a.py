"""
Gate A -- single agent, no marks, sweep poison_delay in {5, 25, 50}.

Question: does a lone agent learn to avoid berry 0 at all (environment-grounded
avoidance, the positive control), and does that avoidance degrade as the delay
between eating and the penalty grows?

Metric: eat0 (poison berries eaten per episode) over training. Run:
    python gate_a.py
"""
from rppo import run_grid

DELAYS = [5, 25, 50]
SEEDS = [0, 1, 2, 3, 4]
ITERS = 600
T = 500

if __name__ == "__main__":
    cells = [dict(marked_berries=(), poison_delay=d, n_agents=1,
                  seed=s, iters=ITERS, T=T)
             for d in DELAYS for s in SEEDS]
    run_grid(cells, "gate_a.csv")
