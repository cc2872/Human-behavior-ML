"""
berryworld_jax.py -- pure-JAX port of berryworld.py, diffed against the NumPy
oracle (berryworld.BerryWorld). Fixed array shapes, no python control flow in
the stepped path, so it jits and vmaps over seeds/agents on device.

Design (see project_brief.md 3.1, 7):
  * Patch construction stays HOST-side (NumPy oracle builds patch_mask); the
    JAX env takes it as a static-shaped input. The hard combinatorial part is
    not re-derived here.
  * Removal/respawn are MASKS (active, respawn timer), never reshapes.
  * The order-dependent zap (NumPy mutates `alive` as targets fall) is
    replicated with lax.scan over agents in index order, so beam resolution
    matches the oracle rather than approximating it.

Correctness is established by oracle diff, not by re-reading this file: see
diff_jax_oracle.py. Deterministic dynamics match exactly; stochastic draws
(regrowth, respawn placement) use JAX PRNG and are checked distributionally.
"""
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

N_ACTIONS = 7
_DELTA = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])   # N, E, S, W


class JCfg(NamedTuple):
    """Static config (python scalars -> hashable -> static_argnums)."""
    grid: int = 15
    n_agents: int = 6
    view: int = 3
    n_berry_types: int = 2
    poison_delay: int = 25
    mark_steps: int = 40
    zap_range: int = 4
    zap_removal_steps: int = 25
    episode_len: int = 300
    r_eat: float = 1.0
    r_poison: float = 4.0
    c_zap: float = 0.1
    c_zapped: float = 2.0
    r_zap_bonus: float = 0.0
    regrow_prob: float = 0.01
    marked_mask: tuple = (True, True)     # per-type: does eating it show a mark
    # observe_pending: Gate-1 confound probe (project_brief.md 4). True = the
    # "poison incoming" flag is visible in the observation; False (default) =
    # hidden, so avoidance must be credited to the eat. Default is bit-exact.
    observe_pending: bool = False
    # bonus_requires_mark: Koster's zap reward is mark-CONTINGENT (paid for zapping
    # a MARKED/norm-violating target, not an unmarked one) -- the incentive that
    # makes selective enforcement individually rational. Default False = flat bonus
    # for ANY landed zap (bit-exact with the pre-flag env); True = pay r_zap_bonus
    # only when the target is marked. Fixes the fidelity gap that flattened
    # selectivity at every bonus (see Koster Fig 4 enforcement economy).
    bonus_requires_mark: bool = False


class State(NamedTuple):
    berries: jnp.ndarray      # (T, G, G) bool
    pos: jnp.ndarray          # (N, 2) int32
    facing: jnp.ndarray       # (N,) int32
    marks: jnp.ndarray        # (N, T) int32   countdown
    pending: jnp.ndarray      # (N, D+1) bool
    respawn: jnp.ndarray      # (N,) int32     0 = active
    patch_mask: jnp.ndarray   # (T, G, G) bool  static regrow template
    t: jnp.ndarray            # scalar int32
    key: jnp.ndarray          # PRNG key


def _cell_key(pos, G):
    return pos[:, 0] * G + pos[:, 1]


@partial(jax.jit, static_argnums=(0,))
def observe(cfg: JCfg, s: State):
    """Egocentric (2v+1)^2 windows, channels [wall, berry0, berry1, agent,
    mark0, mark1] + [facing, pending(zeroed), mark0>0, mark1>0]."""
    G, v, T, N = cfg.grid, cfg.view, cfg.n_berry_types, cfg.n_agents
    w = 2 * v + 1
    active = s.respawn == 0

    wall = jnp.zeros((G, G), jnp.float32)
    wall = wall.at[0, :].set(1.).at[-1, :].set(1.).at[:, 0].set(1.).at[:, -1].set(1.)

    occ = jnp.zeros((G, G), jnp.float32)
    occ = occ.at[s.pos[:, 0], s.pos[:, 1]].add(active.astype(jnp.float32))

    marked_mask = jnp.array(cfg.marked_mask)                 # (T,)
    vis = (s.marks > 0) & marked_mask[None, :] & active[:, None]   # (N, T)
    mk = jnp.zeros((T, G, G), jnp.float32)
    for t in range(T):
        mk = mk.at[t, s.pos[:, 0], s.pos[:, 1]].add(vis[:, t].astype(jnp.float32))

    planes = jnp.concatenate(
        [wall[None], s.berries.astype(jnp.float32), occ[None], mk], 0)   # (P,G,G)
    P = planes.shape[0]
    pad = jnp.pad(planes, ((0, 0), (v, v), (v, v)))

    def window(p):                                           # p = (r, q)
        return lax.dynamic_slice(pad, (0, p[0], p[1]), (P, w, w))
    wins = jax.vmap(window)(s.pos).reshape(N, -1)            # (N, P*w*w)

    # pending slot: hidden by default (bit-exact zeros); when observe_pending is
    # set, expose a binary "poison incoming" flag -- the brief-4 confound probe.
    pending_feat = (s.pending.any(1)[:, None].astype(jnp.float32)
                    if cfg.observe_pending
                    else jnp.zeros((N, 1), jnp.float32))
    self_feats = jnp.concatenate([
        (s.facing / 3.0)[:, None],
        pending_feat,
        (s.marks > 0).astype(jnp.float32),
    ], axis=1)                                               # (N, 2+T)
    obs = jnp.concatenate([wins, self_feats], axis=1)
    return obs * active[:, None].astype(jnp.float32)         # removed -> zero obs


def reset(cfg: JCfg, patch_mask, pos, facing, key):
    """Deterministic-init reset: caller supplies patch_mask, pos, facing (so it
    can mirror the NumPy oracle exactly). Berries start = patch_mask."""
    N, T, D = cfg.n_agents, cfg.n_berry_types, cfg.poison_delay
    s = State(
        berries=jnp.asarray(patch_mask, bool),
        pos=jnp.asarray(pos, jnp.int32),
        facing=jnp.asarray(facing, jnp.int32),
        marks=jnp.zeros((N, T), jnp.int32),
        pending=jnp.zeros((N, D + 1), bool),
        respawn=jnp.zeros(N, jnp.int32),
        patch_mask=jnp.asarray(patch_mask, bool),
        t=jnp.int32(0),
        key=key)
    return s, observe(cfg, s)


@partial(jax.jit, static_argnums=(0,))
def step(cfg: JCfg, s: State, actions, enforce=True):
    # enforce: Phase-2 ghost flag. True = normal (punishment on). False = ghost
    # cell: a landed zap still fires, costs the zapper, pays r_zap_bonus, and is
    # visible as a beam -- but inflicts NO penalty and NO removal on the target.
    # Gates ONLY the two target-side terms below, so enforce=True is bit-exact
    # with the pre-flag env (x*1.0==x, x&True==x). See project_brief.md 3.3.
    G, N, T = cfg.grid, cfg.n_agents, cfg.n_berry_types
    actions = jnp.asarray(actions, jnp.int32)
    rew = jnp.zeros(N, jnp.float32)
    enf_f = jnp.asarray(enforce, jnp.float32)      # 1.0 = punish targets
    enf_b = jnp.asarray(enforce, bool)             # False = ghost (no removal)

    # --- 1. delayed poison lands first
    poison_hits = s.pending[:, 0]
    rew = rew - cfg.r_poison * poison_hits.astype(jnp.float32)
    pending = jnp.concatenate([s.pending[:, 1:], jnp.zeros((N, 1), bool)], 1)

    active = s.respawn == 0

    # --- 2. movement, simultaneous; collisions cancel; removed excluded
    mv = (actions < 4) & active
    facing = jnp.where(mv, actions, s.facing)
    step_vec = _DELTA[jnp.clip(actions, 0, 3)] * mv[:, None]
    target = jnp.clip(s.pos + step_vec, 1, G - 2)
    keys = _cell_key(target, G)                             # (N,)
    # occupancy count over ACTIVE agents' target cells (inactive don't block)
    occ_count = jnp.zeros(G * G, jnp.int32).at[keys].add(active.astype(jnp.int32))
    unique = occ_count[keys] == 1
    ok = jnp.where(active, unique, True)                   # inactive: no-op move
    pos = jnp.where(ok[:, None], target, s.pos)

    # --- 3. eating (positions are distinct after collision -> no conflict)
    cell_berry = s.berries[:, pos[:, 0], pos[:, 1]].T       # (N, T) berry at each pos
    eat = (actions == 4) & active
    has_here = cell_berry.any(1) & eat
    eaten_t = jnp.argmax(cell_berry, axis=1)                # first True type
    did_eat = has_here
    rew = rew + cfg.r_eat * did_eat.astype(jnp.float32)
    onehot = jax.nn.one_hot(eaten_t, T, dtype=jnp.int32) * did_eat[:, None]
    marks = jnp.maximum(s.marks, onehot * cfg.mark_steps)
    # remove eaten berries (distinct positions -> scatter has no collision)
    berries = s.berries
    berries = berries.at[eaten_t, pos[:, 0], pos[:, 1]].set(
        jnp.where(did_eat, False, berries[eaten_t, pos[:, 0], pos[:, 1]]))
    # queue poison for type-0 eats
    ate0 = did_eat & (eaten_t == 0)
    pending = pending.at[:, -1].set(pending[:, -1] | ate0)

    # --- 4. zapping: scan agents in index order, mutating `alive` (order-exact)
    # vis_marked uses POST-eating marks (oracle computes it after step 3), so a
    # berry eaten this step can be enforced this same step.
    zap = (actions == 5) & active
    marked_mask = jnp.array(cfg.marked_mask)
    vis_marked = ((marks > 0) & marked_mask[None, :]).any(1)     # (N,)

    def zap_one(carry, i):
        alive, rew_c, respawn_c, n_land, n_marked = carry
        d = _DELTA[facing[i]]

        def scan_beam(bcarry, k):
            found, tgt = bcarry
            cell = pos[i] + d * (k + 1)
            hitmask = alive & (pos[:, 0] == cell[0]) & (pos[:, 1] == cell[1])
            any_hit = hitmask.any() & (~found)
            first = jnp.argmax(hitmask)                     # first alive at cell
            tgt = jnp.where(any_hit, first, tgt)
            found = found | (hitmask.any())
            return (found, tgt), None
        (found, tgt), _ = lax.scan(scan_beam, (False, 0), jnp.arange(cfg.zap_range))

        fires = zap[i]
        landed = fires & found
        rew_c = rew_c - jax.nn.one_hot(i, N) * (cfg.c_zap * fires)  # cost to zapper i
        # mark-contingent bonus (Koster fidelity). cfg is static, so the off-branch
        # is the exact original expression -> bit-exact when the flag is False.
        bonus_landed = (landed & vis_marked[tgt]) if cfg.bonus_requires_mark else landed
        rew_c = rew_c + jax.nn.one_hot(i, N) * (cfg.r_zap_bonus * bonus_landed)
        rew_c = rew_c - jax.nn.one_hot(tgt, N) * (cfg.c_zapped * landed * enf_f)
        remove = landed & (cfg.zap_removal_steps > 0) & enf_b
        respawn_c = jnp.where(jax.nn.one_hot(tgt, N, dtype=bool) & remove,
                              cfg.zap_removal_steps, respawn_c)
        alive = alive & ~(jax.nn.one_hot(tgt, N, dtype=bool) & remove)
        n_land = n_land + landed.astype(jnp.int32)
        n_marked = n_marked + (landed & vis_marked[tgt]).astype(jnp.int32)
        return (alive, rew_c, respawn_c, n_land, n_marked), None

    (alive, rew, respawn, zaps_landed, zaps_on_marked), _ = lax.scan(
        zap_one, (active, rew, s.respawn, jnp.int32(0), jnp.int32(0)),
        jnp.arange(N))

    # --- 5. regrowth
    key, kg = jax.random.split(s.key)
    grow = (s.patch_mask & ~berries) & (
        jax.random.uniform(kg, berries.shape) < cfg.regrow_prob)
    berries = berries | grow

    # --- 5b. respawn: tick timers; place respawned agents at free interior cells
    respawning = respawn == 1
    respawn = jnp.maximum(respawn - 1, 0)
    key, kr = jax.random.split(key)
    on_grid = (respawn == 0) & ~respawning
    occ_after = jnp.zeros((G, G), bool).at[pos[:, 0], pos[:, 1]].max(on_grid)
    interior = jnp.zeros((G, G), bool).at[1:G - 1, 1:G - 1].set(True)
    free = interior & ~berries.any(0) & ~occ_after          # (G,G) bool
    # pick, for each respawning agent, a distinct free cell via gumbel argmax
    flat_free = free.reshape(-1)
    def place_one(carry, i):
        taken, key_c, pos_c = carry
        key_c, ksub = jax.random.split(key_c)
        avail = flat_free & ~taken
        g = jax.random.gumbel(ksub, (G * G,)) + jnp.where(avail, 0., -1e9)
        cidx = jnp.argmax(g)
        newpos = jnp.array([cidx // G, cidx % G], jnp.int32)
        do = respawning[i]
        pos_c = pos_c.at[i].set(jnp.where(do, newpos, pos_c[i]))
        taken = taken.at[cidx].set(taken[cidx] | do)
        return (taken, key_c, pos_c), None
    (_, key, pos), _ = lax.scan(
        place_one, (jnp.zeros(G * G, bool), kr, pos), jnp.arange(N))
    key, kf = jax.random.split(key)
    facing = jnp.where(respawning, jax.random.randint(kf, (N,), 0, 4), facing)

    # --- 6. mark decay, clock
    marks = jnp.maximum(marks - 1, 0)
    t = s.t + 1
    ns = State(berries, pos, facing, marks, pending, respawn, s.patch_mask, t, key)
    done = t >= cfg.episode_len
    eats = jnp.array([jnp.sum(did_eat & (eaten_t == k)) for k in range(T)])
    info = dict(eats=eats,
                zaps_fired=jnp.sum(zap), zaps_landed=zaps_landed,
                zaps_on_marked=zaps_on_marked, poison_hits=jnp.sum(poison_hits),
                marked_agents=jnp.sum(vis_marked & active),
                active_agents=jnp.sum(active))
    return ns, observe(cfg, ns), rew, done, info
