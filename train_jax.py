"""
train_jax.py -- recurrent IPPO in JAX for berryworld_jax, fully jitted so it
runs on device and vmaps over seeds. Per-agent INDEPENDENT parameters (stacked
leading dim = pool size), so removing an agent is dropping a slice.

Validation discipline: at N=1, marked=(), this must reproduce Gate A (a lone
agent learns to avoid the poison berry). If it can't reproduce a result we
already have on CPU/PyTorch, the port is wrong -- don't spend GPU on it.

    python train_jax.py           # N=1 Gate A smoke on CPU
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
import optax

import berryworld_jax as bwj
from berryworld import BerryWorld, Config


# ------------------------------------------------------------------- network
class ACGRU(nn.Module):
    hidden: int
    n_actions: int

    @nn.compact
    def __call__(self, carry, obs):
        x = nn.tanh(nn.Dense(self.hidden)(obs))
        carry, h = nn.GRUCell(features=self.hidden)(carry, x)
        logits = nn.Dense(self.n_actions)(h)
        val = nn.Dense(1)(h)[..., 0]
        return carry, logits, val


# ------------------------------------------------------------- jittable reset
def _place(cfg, patch_mask, key):
    """Sample N distinct free interior cells (jittable, gumbel-masked)."""
    G, N = cfg.grid, cfg.n_agents
    interior = jnp.zeros((G, G), bool).at[1:G - 1, 1:G - 1].set(True)
    free = (interior & ~jnp.asarray(patch_mask).any(0)).reshape(-1)

    def pick(carry, _):
        taken, k = carry
        k, ks = jax.random.split(k)
        g = jax.random.gumbel(ks, (G * G,)) + jnp.where(free & ~taken, 0., -1e9)
        c = jnp.argmax(g)
        taken = taken.at[c].set(True)
        return (taken, k), c
    (_, _), cells = lax.scan(pick, (jnp.zeros(G * G, bool), key), None, length=N)
    pos = jnp.stack([cells // G, cells % G], axis=1).astype(jnp.int32)
    return pos


def reset_env(cfg, patch_mask, key):
    kp, kf, ks = jax.random.split(key, 3)
    pos = _place(cfg, patch_mask, kp)
    facing = jax.random.randint(kf, (cfg.n_agents,), 0, 4)
    s, obs = bwj.reset(cfg, jnp.asarray(patch_mask), pos, facing, ks)
    return s, obs


# --------------------------------------------------------------------- train
def make_train(cfg, patch_mask, hp):
    N, E = cfg.n_agents, hp["num_envs"]
    net = ACGRU(hp["hidden"], bwj.N_ACTIONS)
    obs_dim = 6 * (2 * cfg.view + 1) ** 2 + 2 + cfg.n_berry_types
    tx = optax.chain(optax.clip_by_global_norm(hp["max_grad"]),
                     optax.adam(hp["lr"]))

    vstep = jax.vmap(lambda s, a: bwj.step(cfg, s, a))       # over envs
    vreset = jax.vmap(lambda k: reset_env(cfg, patch_mask, k))

    def agent_apply(params, carry, obs):                    # obs (E,D) carry (E,H)
        return net.apply(params, carry, obs)
    # over agents: params axis 0, carry axis 0, obs axis 1(agent) -> outputs (N,E,*)
    fwd = jax.vmap(agent_apply, in_axes=(0, 0, 1))

    def train(rng):
        rng, ki = jax.random.split(rng)
        c0 = jnp.zeros((E, hp["hidden"]))
        o0 = jnp.zeros((E, obs_dim))
        params = jax.vmap(lambda k: net.init(k, c0, o0))(
            jax.random.split(ki, N))
        opt_state = tx.init(params)

        def update(runner, _):
            params, opt_state, rng = runner
            rng, kr = jax.random.split(rng)
            state, obs = vreset(jax.random.split(kr, E))     # obs (E,N,D)
            carry = jnp.zeros((N, E, hp["hidden"]))

            # --- rollout one episode across E envs via scan over time
            def step_t(carry_all, key_t):
                carry, state, obs = carry_all
                new_carry, logits, val = fwd(params, carry, obs)   # (N,E,*)
                key_t, ksa = jax.random.split(key_t)
                acts = jax.random.categorical(ksa, logits)         # (N,E)
                logp = jnp.take_along_axis(
                    jax.nn.log_softmax(logits), acts[..., None], -1)[..., 0]
                nstate, nobs, rew, done, info = vstep(state, acts.T)  # env wants (E,N)
                # store agent-major (N,E,*); obs came from env as (E,N,D)
                trans = (obs.transpose(1, 0, 2), acts, logp, val, rew.T,
                         info["eats"], info["zaps_landed"],
                         info["zaps_on_marked"], info["marked_agents"],
                         info["active_agents"])
                return (new_carry, nstate, nobs), trans

            rng, kt = jax.random.split(rng)
            (_, state, _), traj = lax.scan(
                step_t, (carry, state, obs),
                jax.random.split(kt, cfg.episode_len))
            (obs_t, act_t, logp_t, val_t, rew_t, eats_t,
             zl_t, zm_t, ma_t, aa_t) = traj                        # (T,N,E,*)

            # --- GAE per (agent, env)
            def gae_scan(carry, x):
                gae, next_v = carry
                rew, val = x
                delta = rew + hp["gamma"] * next_v - val
                gae = delta + hp["gamma"] * hp["lam"] * gae
                return (gae, val), gae
            _, adv = lax.scan(gae_scan, (jnp.zeros((N, E)), jnp.zeros((N, E))),
                              (rew_t, val_t), reverse=True)
            ret = adv + val_t
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # --- PPO update (independent per agent; single optimizer over stack)
            def loss_fn(params):
                def replay_agent(p, obs_a, act_a):            # (T,E,D),(T,E)
                    def rstep(carry, o):
                        carry, logits, val = net.apply(p, carry, o)
                        return carry, (logits, val)
                    _, (logits, val) = lax.scan(
                        rstep, jnp.zeros((E, hp["hidden"])), obs_a)
                    return logits, val
                logits, val = jax.vmap(replay_agent, in_axes=(0, 1, 1))(
                    params, obs_t, act_t)                     # (N,T,E,*)
                logits = logits.transpose(1, 0, 2, 3)         # (T,N,E,A)
                val = val.transpose(1, 0, 2)                  # (T,N,E)
                logp = jnp.take_along_axis(
                    jax.nn.log_softmax(logits), act_t[..., None], -1)[..., 0]
                ratio = jnp.exp(logp - logp_t)
                p1 = ratio * adv
                p2 = jnp.clip(ratio, 1 - hp["clip"], 1 + hp["clip"]) * adv
                pi_loss = -jnp.minimum(p1, p2).mean()
                v_loss = ((val - ret) ** 2).mean()
                probs = jax.nn.softmax(logits)
                ent = -(probs * jax.nn.log_softmax(logits)).sum(-1).mean()
                return pi_loss + hp["vf"] * v_loss - hp["ent"] * ent

            def ppo_epoch(carry, _):
                params, opt_state = carry
                g = jax.grad(loss_fn)(params)
                upd, opt_state = tx.update(g, opt_state, params)
                params = optax.apply_updates(params, upd)
                return (params, opt_state), None
            (params, opt_state), _ = lax.scan(
                ppo_epoch, (params, opt_state), None, length=hp["epochs"])

            # eats_t is (T, E, n_berry_types); per-env episode totals, env-mean
            zl, zm = zl_t.sum(), zm_t.sum()
            prev = ma_t.sum() / jnp.maximum(aa_t.sum(), 1)      # marked prevalence
            share = zm / jnp.maximum(zl, 1)                     # zaps hitting marked
            metrics = dict(eat0=eats_t[..., 0].sum() / E,
                           eat1=eats_t[..., 1].sum() / E,
                           ret=rew_t.sum(0).mean(),
                           selectivity=share / jnp.maximum(prev, 1e-8),
                           prevalence=prev,                     # marked frac -> ceiling=1/prev
                           zaps=zl / E)                         # landed zaps/episode
            return (params, opt_state, rng), metrics

        (params, _, _), metrics = lax.scan(
            update, (params, opt_state, rng), None, length=hp["updates"])
        return params, metrics

    return train


DEFAULT_HP = dict(hidden=64, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                  epochs=3, ent=0.01, vf=0.5, max_grad=0.5,
                  num_envs=16, updates=400)


def build_patch_mask(marked, n_agents, seed=0):
    c = Config(); c.marked_berries = marked; c.n_agents = n_agents
    return BerryWorld(c, seed=seed).patch_mask.copy()


if __name__ == "__main__":
    import time
    marked = ()
    cfg = bwj.JCfg(n_agents=1, episode_len=300, poison_delay=25,
                   zap_removal_steps=25,
                   marked_mask=tuple(t in marked for t in range(2)))
    pm = build_patch_mask(marked, 1)
    train = make_train(cfg, pm, DEFAULT_HP)
    t0 = time.time()
    params, metrics = jax.block_until_ready(jax.jit(train)(jax.random.PRNGKey(0)))
    dt = time.time() - t0
    e0 = np.array(metrics["eat0"])
    print(f"Gate A (N=1, no marks) -- {dt:.1f}s for {DEFAULT_HP['updates']} updates")
    print(f"  poison eaten/episode: first10 {e0[:10].mean():.1f} -> last10 {e0[-10:].mean():.1f}")
    print("  PASS (avoidance learned)" if e0[-10:].mean() < 0.6 * e0[:10].mean()
          else "  (no clear avoidance -- inspect)")
