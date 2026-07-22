"""
Gate C -- full 6-agent population, poison_delay=25, three conditions:
    ()     no rule          marked_berries=()
    (0,)   important rule    marked_berries=(0,)
    (0,1)  important + silly marked_berries=(0, 1)

Question: is poison-avoidance learned FASTER in (0,) than in ()? The mark plus
learned zapping give a faster social signal than the delayed poison alone.

Metric: eat0 (population poison berries eaten per episode) over training. Run:
    python gate_c.py
"""
from rppo import run_grid

CONDITIONS = [(), (0,), (0, 1)]
SEEDS = [0, 1, 2, 3, 4]
ITERS = 400
T = 300

if __name__ == "__main__":
    cells = [dict(marked_berries=c, poison_delay=25, n_agents=6,
                  seed=s, iters=ITERS, T=T)
             for c in CONDITIONS for s in SEEDS]
    run_grid(cells, "gate_c.csv")
