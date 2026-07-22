"""
berryworld.py -- minimal reference implementation of a Koster-style norm substrate.

NumPy, single file, no dependencies beyond numpy. This is the *debuggable*
version: written for clarity so the game logic can be verified before porting
to JAX. Every array has a fixed shape so the port is mechanical.

MECHANICS
---------
Two berry types on a grid.
  Berry 0 ("poison"): +r_eat now, then -r_poison delayed by D steps.
                      Grounded in the ENVIRONMENT. Avoidance should survive
                      isolation -- this is the positive control.
  Berry 1 ("harmless"): +r_eat now, no consequence ever.
                        Grounded only in the POPULATION, if marked.

Eating berry type t sets a visible mark of type t on the eater for M steps.
Which berry types produce a visible mark is a CONDITION parameter
(`marked_berries`), giving Koster's three conditions:
    ()      -> no rule
    (0,)    -> important rule only
    (0, 1)  -> important + silly rule

Zapping costs the zapper c_zap and the target c_zapped. The environment
does NOT know which marks "deserve" punishment. Who gets zapped is learned.
That is the whole point -- do not build the norm in.

Observations are egocentric windows, flattened. Rewards are per-agent.

SMOKE TEST
----------
  python berryworld.py
"""
import numpy as np

# ------------------------------------------------------------------ config
class Config:
    grid = 15            # square grid, walls on the border
    n_agents = 6
    view = 3             # egocentric half-window -> (2*view+1)^2 cells
    n_berry_types = 2

    cells_per_type = 54  # berry cells per type, equal by construction so
                         # scarcity can't bias consumption. 54 = 6 disjoint
                         # 3x3 blocks, ~ the original type-0 poison pressure.
    # Outcome-neutral BATCH-check thresholds (see outcome_neutral_suite).
    # Per-seed clustering gaps of the fixed and old constructions OVERLAP
    # (fixed |gap| up to 0.41, old down to 0.04), so no per-seed threshold has
    # power. The powered test is on the ACROSS-SEED MEAN gap: null (fixed) is
    # 0.036 +/- 0.030 SEM over 30 seeds; the broken construction sits at 0.50.
    clustering_gap_tol = 0.15  # ~4x null SEM: passes fixed, rejects old (~8 SEM)
    eats_t_tol = 3.0           # |paired t| on eats-by-type under random policy
    val_seeds = 30             # seeds for the batch outcome-neutral suite
    regrow_prob = 0.01   # per-step per-empty-site regrowth within a patch

    r_eat = 1.0
    r_poison = 4.0       # magnitude of the delayed penalty
    poison_delay = 25    # D: steps between eating berry 0 and the penalty
    mark_steps = 40      # M: how long a mark stays visible

    c_zap = 0.1          # cost to the zapper
    c_zapped = 2.0       # cost to the target
    zap_range = 4        # beam length, fires along facing direction

    marked_berries = (0, 1)   # condition parameter; see docstring
    episode_len = 1000


# actions: 0-3 move N/E/S/W (also sets facing), 4 eat, 5 zap, 6 noop
N_ACTIONS = 7
_DELTA = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])


class BerryWorld:
    def __init__(self, cfg=Config(), seed=0):
        self.c = cfg
        self.rng = np.random.default_rng(seed)
        self._build_patches()
        self.reset()

    # ------------------------------------------------------------- setup
    @staticmethod
    def _clustering(mask):
        """Mean number of 4-neighbours of each berry cell that share its type.
        An isolated 3x3 block scores 24/9 = 2.67; fragmented remnants score
        lower, touching/overlapping blocks higher. Used to check that the two
        types are structurally comparable, not just equal in count."""
        nb = np.zeros(mask.shape, int)
        nb[1:, :] += mask[:-1, :]
        nb[:-1, :] += mask[1:, :]
        nb[:, 1:] += mask[:, :-1]
        nb[:, :-1] += mask[:, 1:]
        n = int(mask.sum())
        return float(nb[mask].sum()) / n if n else 0.0

    def _build_patches(self):
        """Symmetric patch construction, resampled only on __init__ so
        episodes are comparable across a run. Both berry types are drawn from
        ONE generative process -- alternate type, place a random 3x3 block
        whose footprint is disjoint from every already-claimed cell -- so
        spatial structure (both count AND clustering) is identical by symmetry
        rather than by post-hoc correction. The old "earlier type wins the
        overlap" rule made later types both scarcer and more fragmented, with
        seed-dependent variance; this removes both confounds at the source."""
        c = self.c
        G = c.grid
        T = c.n_berry_types
        target = c.cells_per_type
        self.patch_mask = np.zeros((T, G, G), bool)
        claimed = np.zeros((G, G), bool)        # any-type occupancy
        centres = [[] for _ in range(T)]        # 3x3 block centres per type

        # Phase 1: disjoint 3x3 blocks, alternating type. Centres kept one cell
        # in from the interior edge so each block lands fully inside -> exactly
        # 9 cells, no border clipping. 12 disjoint blocks (target 54) is above
        # the random-packing (RSA) jamming limit for this grid, so one type may
        # place fewer than the other -- Phase 2 equalizes that away.
        want = target // 9                      # full blocks desired per type
        order, stalled = 0, 0
        while any(len(centres[t]) < want for t in range(T)) and stalled < T:
            t = order % T
            order += 1
            if len(centres[t]) >= want:
                continue
            for _ in range(500):                # rejection sampling
                r = int(self.rng.integers(2, G - 2))
                q = int(self.rng.integers(2, G - 2))
                blk = (slice(r - 1, r + 2), slice(q - 1, q + 2))
                if not claimed[blk].any():
                    claimed[blk] = True
                    centres[t].append((r, q))
                    stalled = 0
                    break
            else:                               # no disjoint spot found
                stalled += 1

        # Phase 2: force IDENTICAL composition across types so structure can't
        # depend on placement order. Keep the same number of full blocks for
        # every type (drop extras from whichever type packed more, freeing
        # them), then top every type up to `target` with the same number of
        # loose cells drawn from the shared unclaimed interior pool.
        n_blk = min(len(cs) for cs in centres)
        for t in range(T):
            for (r, q) in centres[t][n_blk:]:   # release surplus blocks
                claimed[r - 1:r + 2, q - 1:q + 2] = False
            for (r, q) in centres[t][:n_blk]:   # paint the kept blocks
                self.patch_mask[t, r - 1:r + 2, q - 1:q + 2] = True
        for t in range(T):
            need = target - int(self.patch_mask[t].sum())
            for _ in range(need):
                free = np.argwhere(~claimed)
                free = free[(free[:, 0] > 0) & (free[:, 0] < G - 1)
                            & (free[:, 1] > 0) & (free[:, 1] < G - 1)]
                if not len(free):
                    break
                r, q = free[int(self.rng.integers(len(free)))]
                self.patch_mask[t, r, q] = True
                claimed[r, q] = True

        # --- exact per-instance structural invariant: equal cells per type.
        # Clustering is a DISTRIBUTIONAL property -- its per-seed gap has no
        # power (fixed and broken constructions overlap seed-to-seed), so it is
        # verified on the across-seed mean in outcome_neutral_suite(), not here.
        counts = self.patch_mask.reshape(c.n_berry_types, -1).sum(1)
        assert (counts == counts[0]).all(), \
            f"invariant violated: unequal cells per type {counts}"
        self.patch_clustering = np.array(
            [self._clustering(self.patch_mask[t])
             for t in range(c.n_berry_types)])  # exposed for the batch check

    def reset(self):
        c = self.c
        self.t = 0
        self.berries = self.patch_mask.copy()          # (T, G, G) bool

        free = np.argwhere(~self.berries.any(0))
        free = free[(free[:, 0] > 0) & (free[:, 0] < c.grid - 1)
                    & (free[:, 1] > 0) & (free[:, 1] < c.grid - 1)]
        idx = self.rng.choice(len(free), c.n_agents, replace=False)
        self.pos = free[idx].copy()                    # (N, 2) int
        self.facing = self.rng.integers(0, 4, c.n_agents)

        self.marks = np.zeros((c.n_agents, c.n_berry_types), int)   # countdown
        # pending[i, k] = steps until the k-th queued poison hit lands, 0 = none
        self.pending = np.zeros((c.n_agents, c.poison_delay + 1), bool)
        return self.observe()

    # ------------------------------------------------------- observation
    def observe(self):
        """Egocentric (2v+1)^2 windows, channels:
        [wall, berry0, berry1, agent, mark0, mark1] + self scalars."""
        c, v = self.c, self.c.view
        w = 2 * v + 1
        G = c.grid

        wall = np.zeros((G, G), np.float32)
        wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = 1.0

        occ = np.zeros((G, G), np.float32)
        mk = np.zeros((c.n_berry_types, G, G), np.float32)
        for i, (r, q) in enumerate(self.pos):
            occ[r, q] = 1.0
            for t in range(c.n_berry_types):
                if self.marks[i, t] > 0 and t in c.marked_berries:
                    mk[t, r, q] = 1.0

        planes = np.concatenate(
            [wall[None], self.berries.astype(np.float32), occ[None], mk], 0)
        P = planes.shape[0]
        pad = np.pad(planes, ((0, 0), (v, v), (v, v)), constant_values=0.0)

        obs = np.zeros((c.n_agents, P * w * w + 2 + c.n_berry_types), np.float32)
        for i, (r, q) in enumerate(self.pos):
            win = pad[:, r:r + w, q:q + w]             # (P, w, w)
            self_feats = np.concatenate([
                [self.facing[i] / 3.0],
                [self.pending[i].any().astype(np.float32)],   # NOT observable
                (self.marks[i] > 0).astype(np.float32),
            ])
            obs[i] = np.concatenate([win.ravel(), self_feats])
        # zero the pending channel: the agent must infer poisoning from
        # consequences, not read it off the observation. Flip this to 1.0
        # only as a diagnostic.
        obs[:, -1 - self.c.n_berry_types] = 0.0
        return obs

    # ------------------------------------------------------------- step
    def step(self, actions):
        c = self.c
        actions = np.asarray(actions, int)
        rew = np.zeros(c.n_agents, np.float32)
        info = {"eats": np.zeros(c.n_berry_types, int),  # eats by berry type
                "zaps_fired": 0,    # zap actions issued (each costs c_zap)
                "zaps_landed": 0,   # beams that actually hit an agent
                "poison_hits": 0}   # delayed poison penalties that landed

        # --- 1. delayed poison lands first (independent of this step's action)
        info["poison_hits"] = int(self.pending[:, 0].sum())
        rew -= c.r_poison * self.pending[:, 0]
        self.pending = np.roll(self.pending, -1, axis=1)
        self.pending[:, -1] = False

        # --- 2. movement, resolved simultaneously; collisions cancel
        mv = actions < 4
        self.facing = np.where(mv, actions, self.facing)
        target = self.pos.copy()
        target[mv] += _DELTA[actions[mv]]
        target = np.clip(target, 1, c.grid - 2)
        # reject moves onto a cell another agent already occupies or targets
        keys = target[:, 0] * c.grid + target[:, 1]
        _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        ok = counts[inv] == 1
        self.pos[ok] = target[ok]

        # --- 3. eating
        eat = actions == 4
        for i in np.flatnonzero(eat):
            r, q = self.pos[i]
            for t in range(c.n_berry_types):
                if self.berries[t, r, q]:
                    self.berries[t, r, q] = False
                    rew[i] += c.r_eat
                    self.marks[i, t] = c.mark_steps
                    info["eats"][t] += 1
                    if t == 0:
                        self.pending[i, -1] = True     # lands in D steps
                    break

        # --- 4. zapping: beam along facing, hits the nearest agent in range
        zap = actions == 5
        info["zaps_fired"] = int(zap.sum())
        for i in np.flatnonzero(zap):
            rew[i] -= c.c_zap
            d = _DELTA[self.facing[i]]
            for k in range(1, c.zap_range + 1):
                cell = self.pos[i] + d * k
                hit = np.flatnonzero(
                    (self.pos[:, 0] == cell[0]) & (self.pos[:, 1] == cell[1]))
                if len(hit):
                    rew[hit[0]] -= c.c_zapped
                    info["zaps_landed"] += 1
                    break

        # --- 5. regrowth and bookkeeping
        empty = self.patch_mask & ~self.berries
        self.berries |= empty & (
            self.rng.random(self.berries.shape) < c.regrow_prob)
        self.marks = np.maximum(self.marks - 1, 0)

        self.t += 1
        done = self.t >= c.episode_len
        return self.observe(), rew, done, info


# ------------------------------------------------- outcome-neutral batch suite
def outcome_neutral_suite(n_seeds=Config.val_seeds, verbose=True):
    """Checks that require a distribution over seeds rather than one
    construction. Pre-specified, orthogonal to any norm/enforcement hypothesis,
    and calibrated to have POWER against the specific failure they guard:

      clustering gap  -- no SYSTEMATIC spatial-structure difference between
                         types. Tested on the across-seed MEAN because the
                         per-seed gap can't separate fixed from broken (their
                         distributions overlap). Null mean 0.036 +/- 0.030 SEM;
                         broken construction sits at ~0.50 -> rejected at ~8 SEM.
      eats by type    -- the DIRECT target the structural checks are proxies
                         for: a null (random) policy must consume both types at
                         indistinguishable rates. Paired t over seeds; unequal
                         counts (the old build) drive |t| far past the tol.

    Returns the computed statistics so they can be logged into the protocol.
    """
    gaps, e0, e1 = [], [], []
    for s in range(n_seeds):
        env = BerryWorld(seed=s)             # equal-count assert fires here
        cl = env.patch_clustering
        gaps.append(cl[0] - cl[1])
        env.reset()
        rng = np.random.default_rng(10_000 + s)
        tot = np.zeros(env.c.n_berry_types, int)
        for _ in range(env.c.episode_len):
            a = rng.integers(0, N_ACTIONS, env.c.n_agents)
            _, _, done, info = env.step(a)
            tot += info["eats"]
            if done:
                break
        e0.append(tot[0]); e1.append(tot[1])

    gaps = np.asarray(gaps, float)
    e0, e1 = np.asarray(e0, float), np.asarray(e1, float)
    diff = e0 - e1
    gap_mean = float(gaps.mean())
    gap_sem = float(gaps.std(ddof=1) / np.sqrt(n_seeds))
    eats_t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n_seeds)))

    if verbose:
        print(f"\n=== outcome-neutral batch suite ({n_seeds} seeds) ===")
        print(f"clustering gap (0-1) mean {gap_mean:+.3f}  SEM {gap_sem:.3f}"
              f"   |tol| {Config.clustering_gap_tol}")
        print(f"eats/type  poison {e0.mean():.1f}  harmless {e1.mean():.1f}"
              f"   paired t {eats_t:+.2f}  |tol| {Config.eats_t_tol}")

    assert abs(gap_mean) < Config.clustering_gap_tol, (
        f"systematic clustering gap {gap_mean:+.3f} "
        f">= tol {Config.clustering_gap_tol}")
    assert abs(eats_t) < Config.eats_t_tol, (
        f"eats-by-type asymmetry under random policy: paired t={eats_t:+.2f}")
    if verbose:
        print("batch checks OK      no systematic clustering gap; "
              "eats-by-type indistinguishable")
    return {"gap_mean": gap_mean, "gap_sem": gap_sem, "eats_t": eats_t,
            "eats_poison": float(e0.mean()), "eats_harmless": float(e1.mean())}


# ------------------------------------------------------------ smoke test
if __name__ == "__main__":
    env = BerryWorld(seed=0)
    obs = env.reset()
    print("obs shape          ", obs.shape)
    print("berries at reset   ", env.berries.sum(axis=(1, 2)))
    print("patch clustering   ", np.round(env.patch_clustering, 2))

    rng = np.random.default_rng(1)
    total = np.zeros(env.c.n_agents)
    eats = np.zeros(env.c.n_berry_types, int)
    zaps_fired = zaps_landed = poison_hits = 0
    for step in range(env.c.episode_len):
        a = rng.integers(0, N_ACTIONS, env.c.n_agents)
        obs, r, done, info = env.step(a)
        eats += info["eats"]
        zaps_fired += info["zaps_fired"]
        zaps_landed += info["zaps_landed"]
        poison_hits += info["poison_hits"]
        total += r
        if done:
            break

    pending_at_end = int(env.pending.sum())

    print("random-policy return", np.round(total, 2))
    print("eats by type        ", eats)
    print("zaps fired / landed ", zaps_fired, "/", zaps_landed)
    print("poison hits landed  ", poison_hits)
    print("poison still pending", pending_at_end)
    print("marks still active  ", (env.marks > 0).sum(0))

    # outcome-neutral BEHAVIOURAL invariant #1: poison bookkeeping. Each
    # type-0 eat queues exactly one delayed hit, so
    #   eats[0] == hits that landed + hits still in flight at episode end.
    assert eats[0] == poison_hits + pending_at_end, (
        f"poison mismatch: {eats[0]} != {poison_hits} + {pending_at_end}")
    print(f"poison check OK      {eats[0]} eaten == "
          f"{poison_hits} landed + {pending_at_end} pending")

    # outcome-neutral BEHAVIOURAL invariant #2: the four reward channels
    # reconstruct the total return with zero residual (no unlogged reward).
    recon = (eats.sum() * env.c.r_eat
             - poison_hits * env.c.r_poison
             - zaps_landed * env.c.c_zapped
             - zaps_fired * env.c.c_zap)
    assert abs(recon - total.sum()) < 1e-4, (
        f"reward residual {recon - total.sum():.4f}: channels don't close")
    print(f"reward check OK      channels reconstruct {total.sum():.1f} "
          f"with 0 residual")

    # structural + behavioural checks that need a distribution over seeds
    outcome_neutral_suite()

    print("\nsanity: return should be near zero or negative under a random")
    print("policy -- eating is rare, zapping is frequent and costly.")
