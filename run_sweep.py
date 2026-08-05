"""
run_sweep.py -- the faithful-scale population sweep, driver for GPU (Colab).

For each (N, condition) cell it vmaps the JAX IPPO train over seeds and logs
per-update metrics. The question this answers: does enforcement (selectivity>1)
fire when the population grows, given enough training?

Local (CPU) smoke:  python run_sweep.py --smoke
Faithful (GPU):     imported from the Colab notebook with a large hp.

NOTE / honest limitations of this diagnostic version:
  * All N agents play every episode (Koster's 8-of-12 per-episode sampling is a
    refinement not yet implemented -- this tests population SIZE, the primary
    lever).
  * patch_mask is fixed per (N,condition) cell across seeds; seed varies net
    init + rollout + resets. Per-seed patches are the rigorous version, later.
"""
import argparse
import csv
import json
import jax
import numpy as np
import train_jax as T
import berryworld_jax as bwj


def env_variant(poison_delay=25, r_zap_bonus=0.0,
                episode_len=300, zap_removal_steps=25, observe_pending=False,
                bonus_requires_mark=False):
    return dict(poison_delay=poison_delay, r_zap_bonus=r_zap_bonus,
                episode_len=episode_len, zap_removal_steps=zap_removal_steps,
                observe_pending=observe_pending,
                bonus_requires_mark=bonus_requires_mark)


def run_sweep(n_list, conditions, n_seeds, hp, env_list, out_csv="sweep.csv",
              vmap_seeds=False):
    """Sweep over N x env_variant x condition. Each env_variant carries its own
    poison_delay (difficulty) and r_zap_bonus (enforcement incentive), both
    logged per row so the D and bonus axes are analysable."""
    if isinstance(env_list, dict):                 # allow a single env
        env_list = [env_list]
    # Guard: refuse a real sweep on CPU. The whole point of the JAX pivot is
    # GPU; a CPU run here means the notebook is executing on the laptop, not
    # Colab. Abort in <1s instead of grinding for hours. Tiny device-sanity
    # runs (updates*num_envs <= 1000) are still allowed on CPU.
    if not any(d.platform == "gpu" for d in jax.devices()) \
            and hp["updates"] * hp["num_envs"] > 1000:
        raise RuntimeError(
            f"run_sweep aborted: NO GPU (jax.devices()={jax.devices()}). "
            f"This is a CPU runtime -- your laptop, not Colab GPU -- and "
            f"updates*num_envs={hp['updates'] * hp['num_envs']} is a real "
            f"sweep (hours on CPU). On Colab: Runtime -> Change runtime type "
            f"-> GPU, then Run all.")
    # Crash-safe: write each cell's rows the instant it finishes and flush, so an
    # interrupt / Colab timeout / preemption keeps every COMPLETED cell instead of
    # losing the whole run (the earlier failure mode -- see project notes on
    # making outputs resumable). `rows` is still accumulated for the return value.
    fieldnames = ["N", "D", "bonus", "condition", "seed", "update",
                  "eat0", "eat1", "selectivity", "prevalence", "zaps", "ret"]
    rows = []
    # Reproducibility: dump the FULL recipe next to the CSV. The pilot proved the
    # CSV's 4 config columns (N/D/bonus/seed) UNDER-specify a run -- num_envs was
    # the difference that flipped the whole result -- so persist hp + env + seeds.
    with open(out_csv + ".config.json", "w") as cf:
        json.dump({"hp": hp, "n_seeds": n_seeds,
                   "conditions": [list(c) for c in conditions],
                   "env_list": [dict(e) for e in env_list],
                   "n_list": list(n_list)}, cf, indent=2)
    f = open(out_csv, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); f.flush()
    try:
        for N in n_list:
            for env in env_list:
                for marked in conditions:
                    cfg = bwj.JCfg(
                        n_agents=N, episode_len=env["episode_len"],
                        poison_delay=env["poison_delay"],
                        zap_removal_steps=env["zap_removal_steps"],
                        r_zap_bonus=env["r_zap_bonus"],
                        observe_pending=env.get("observe_pending", False),
                        bonus_requires_mark=env.get("bonus_requires_mark", False),
                        marked_mask=tuple(t in marked for t in range(2)))
                    pm = T.build_patch_mask(marked, N)
                    train1 = T.make_train(cfg, pm, hp)
                    keys = jax.random.split(jax.random.PRNGKey(0), n_seeds)
                    # Seeds are INDEPENDENT (no cross-seed reduction), so vmapping
                    # them only parallelises -- it does not change per-seed math,
                    # but it holds all n_seeds training graphs on the GPU at once
                    # (~n_seeds x memory, the OOM at large num_envs). Default runs
                    # them SEQUENTIALLY: one seed's graph resident at a time, each
                    # pulled to host so its device buffers free before the next.
                    # vmap_seeds=True restores the parallel path (needs the VRAM;
                    # its XLA codegen differs, so keep it for exact-koster repro).
                    if vmap_seeds:
                        mfn = jax.jit(jax.vmap(lambda k: train1(k)[1]))
                        mm = jax.block_until_ready(mfn(keys))          # each (S, U)
                        m = {kk: np.asarray(vv) for kk, vv in mm.items()}
                    else:
                        seed_metrics = jax.jit(lambda k: train1(k)[1])
                        per_seed = []
                        for k in keys:
                            mk = jax.block_until_ready(seed_metrics(k))
                            per_seed.append({kk: np.asarray(vv)
                                             for kk, vv in mk.items()})
                            del mk
                        m = {kk: np.stack([ps[kk] for ps in per_seed], 0)  # (S, U)
                             for kk in per_seed[0]}
                    cond = "".join(str(b) for b in marked) or "none"
                    U = m["eat0"].shape[1]
                    e = m["eat0"]; sel = m["selectivity"]
                    print(f"N={N:2d} D={env['poison_delay']:2d} "
                          f"bonus={env['r_zap_bonus']:.1f} {cond:4s}  "
                          f"eat0 {e[:, :5].mean():4.0f}->{e[:, -5:].mean():4.0f}"
                          f"  sel {sel[:, -5:].mean():.2f}  (n={n_seeds})", flush=True)
                    cell_rows = [dict(
                        N=N, D=env["poison_delay"],
                        bonus=env["r_zap_bonus"], condition=cond,
                        seed=s, update=u,
                        eat0=float(m["eat0"][s, u]), eat1=float(m["eat1"][s, u]),
                        selectivity=float(m["selectivity"][s, u]),
                        prevalence=float(m["prevalence"][s, u]),
                        zaps=float(m["zaps"][s, u]), ret=float(m["ret"][s, u]))
                        for s in range(n_seeds) for u in range(U)]
                    w.writerows(cell_rows); f.flush()          # persist this cell
                    rows.extend(cell_rows)
    finally:
        f.close()
    print(f"wrote {len(rows)} rows -> {out_csv}", flush=True)
    return rows


ENV = env_variant()                                # D=25, bonus=0 default

# Faithful-scale hyperparameters for GPU. steps = updates * num_envs * episode_len.
# updates=1500, num_envs=256 -> ~1.15e8 steps/run (Koster regime is 2-4e8).
FAITHFUL_HP = dict(hidden=64, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                   epochs=3, ent=0.01, vf=0.5, max_grad=0.5,
                   num_envs=256, updates=1500)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run")
    a = ap.parse_args()
    if a.smoke:
        hp = dict(FAITHFUL_HP); hp["num_envs"] = 8; hp["updates"] = 20
        envs = [env_variant(poison_delay=d) for d in (25, 75)]     # D axis
        run_sweep([12], [(), (0,), (0, 1)], n_seeds=2, hp=hp, env_list=envs,
                  out_csv="sweep_smoke.csv")
    else:
        run_sweep([10, 12], [(), (0,), (0, 1)], n_seeds=8,
                  hp=FAITHFUL_HP, env_list=[ENV], out_csv="sweep.csv")
