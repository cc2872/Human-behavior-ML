"""
rppo.py -- minimal recurrent PPO for the BerryWorld Phase-1 smoke test.

Throwaway diagnostic, not the final apparatus. One GRU actor-critic PER AGENT
(independent weights, so dropping an agent later is a coherent operation).
Single-episode-per-update PPO, no minibatching, no hyperparameter search. Its
only job is direction-of-effect.

The rollout is stepped (actions feed the env); the PPO update replays each
trajectory through nn.GRU in one shot (fast, no python BPTT loop).
"""
import csv
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from berryworld import BerryWorld, Config, N_ACTIONS

torch.set_num_threads(max(1, torch.get_num_threads()))

# fixed hyperparameters -- deliberately not tuned
HID = 64
LR = 3e-4
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
EPOCHS = 3
ENT_COEF = 0.01
V_COEF = 0.5
MAX_GRAD = 0.5


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=__file__.rsplit("\\", 1)[0], text=True).strip()
    except Exception:
        return "nogit"


class RecurrentAC(nn.Module):
    """Per-agent GRU actor-critic over the flat 298-d observation."""

    def __init__(self, obs_dim, n_act, hid=HID):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh())
        self.gru = nn.GRU(hid, hid)                 # (seq, batch, feat)
        self.pi = nn.Linear(hid, n_act)
        self.v = nn.Linear(hid, 1)
        self.hid = hid

    def h0(self):
        return torch.zeros(1, 1, self.hid)

    def step(self, obs_vec, h):
        """One timestep for rollout. obs_vec: (obs_dim,) -> logits, value, h."""
        x = self.enc(obs_vec).view(1, 1, -1)
        y, h = self.gru(x, h)
        y = y.view(-1)
        return self.pi(y), self.v(y).squeeze(-1), h

    def seq(self, obs_seq):
        """Whole trajectory from h=0 in one call. obs_seq: (T, obs_dim)."""
        x = self.enc(obs_seq).unsqueeze(1)          # (T, 1, hid)
        y, _ = self.gru(x, self.h0())               # (T, 1, hid)
        y = y.squeeze(1)                            # (T, hid)
        return self.pi(y), self.v(y).squeeze(-1)


def _gae(rews, vals):
    """GAE with zero bootstrap (episodes terminate)."""
    T = len(rews)
    adv = torch.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nextv = vals[t + 1] if t + 1 < T else 0.0
        delta = rews[t] + GAMMA * nextv - vals[t]
        last = delta + GAMMA * LAM * last
        adv[t] = last
    return adv


def _ppo_update(net, opt, obs, acts, old_logp, vals, rews):
    adv = _gae(rews, vals)
    ret = adv + vals
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(EPOCHS):
        logits, newv = net.seq(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(acts)
        ratio = torch.exp(logp - old_logp)
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv
        pi_loss = -torch.min(s1, s2).mean()
        v_loss = ((newv - ret) ** 2).mean()
        loss = pi_loss + V_COEF * v_loss - ENT_COEF * dist.entropy().mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD)
        opt.step()


def train_run(marked_berries, poison_delay, n_agents, seed, iters, T):
    """One (condition x seed) run. Returns a list of per-iteration row dicts."""
    cfg = Config()
    cfg.marked_berries = marked_berries
    cfg.poison_delay = poison_delay
    cfg.n_agents = n_agents
    cfg.episode_len = T
    env = BerryWorld(cfg, seed=seed)
    obs_dim = env.observe().shape[1]

    torch.manual_seed(seed)
    nets = [RecurrentAC(obs_dim, N_ACTIONS) for _ in range(n_agents)]
    opts = [torch.optim.Adam(n.parameters(), lr=LR) for n in nets]

    sha = git_sha()
    cond = "".join(str(b) for b in marked_berries) or "none"
    rows = []
    env_step = 0
    for it in range(iters):
        obs = env.reset()
        h = [n.h0() for n in nets]
        O = [[] for _ in range(n_agents)]
        A = [[] for _ in range(n_agents)]
        LP = [[] for _ in range(n_agents)]
        V = [[] for _ in range(n_agents)]
        R = [[] for _ in range(n_agents)]
        eat0 = eat1 = 0
        ep_ret = np.zeros(n_agents)
        for t in range(T):
            actions = []
            with torch.no_grad():
                for i, net in enumerate(nets):
                    ov = torch.as_tensor(obs[i], dtype=torch.float32)
                    logits, val, h[i] = net.step(ov, h[i])
                    dist = Categorical(logits=logits)
                    act = dist.sample()
                    O[i].append(obs[i]); A[i].append(int(act))
                    LP[i].append(float(dist.log_prob(act))); V[i].append(float(val))
                    actions.append(int(act))
            obs, rew, done, info = env.step(actions)
            for i in range(n_agents):
                R[i].append(float(rew[i]))
            eat0 += int(info["eats"][0]); eat1 += int(info["eats"][1])
            ep_ret += rew
            env_step += 1
            if done:
                break

        for i, net in enumerate(nets):
            _ppo_update(
                net, opts[i],
                torch.as_tensor(np.asarray(O[i]), dtype=torch.float32),
                torch.as_tensor(A[i]),
                torch.as_tensor(LP[i]),
                torch.as_tensor(V[i]),
                torch.as_tensor(R[i]))

        rows.append(dict(sha=sha, condition=cond, poison_delay=poison_delay,
                         n_agents=n_agents, T=T, seed=seed, iter=it,
                         env_step=env_step, ret=round(float(ep_ret.mean()), 3),
                         eat0=eat0, eat1=eat1))
    return rows


def _worker(cell):
    """Run one cell single-threaded (so many run in parallel across cores)."""
    torch.set_num_threads(1)
    t0 = time.time()
    rows = train_run(cell["marked_berries"], cell["poison_delay"],
                     cell["n_agents"], cell["seed"], cell["iters"], cell["T"])
    return rows, (time.time() - t0) / 60.0


def run_grid(cells, out_csv, budget_min=30.0, workers=8):
    """cells: list of dicts with keys marked_berries, poison_delay, n_agents,
    seed, iters, T. Runs them in parallel (one single-threaded process each),
    writes all rows to out_csv, flags any run over budget_min."""
    import concurrent.futures as cf
    fields = ["sha", "condition", "poison_delay", "n_agents", "T", "seed",
              "iter", "env_step", "ret", "eat0", "eat1"]
    all_rows = []
    done = 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, c): c for c in cells}
        for fut in cf.as_completed(futs):
            rows, dt = fut.result()
            all_rows.extend(rows)
            done += 1
            flag = "  <== EXCEEDS BUDGET" if dt > budget_min else ""
            print(f"[{done:2d}/{len(cells)}] cond={rows[0]['condition']:4s} "
                  f"D={rows[0]['poison_delay']:2d} N={rows[0]['n_agents']} "
                  f"seed={rows[0]['seed']}  {dt:5.1f} min  "
                  f"eat0 {rows[0]['eat0']}->{rows[-1]['eat0']}{flag}", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {out_csv}")
