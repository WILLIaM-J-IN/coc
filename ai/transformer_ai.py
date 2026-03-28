"""
transformer_rl.py
=================
Transformer RL agent for CoC battle automation.

Click area enforcement:
  - agent.window_rect = (left, top, w, h)  ← set by main.py via WindowArea
  - All output coordinates are scaled within this rectangle
  - window_helper.clamp_game / clamp_map applied as final safety net

  template 2-10  → agent receives game_rect  → clicks within game area
  template 11-15 → agent receives map_rect   → clicks within map area

Requirements:
    pip install torch torchvision
"""

import os
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torchvision import transforms
from PIL import Image


# ============================================================
# Config
# ============================================================

IMG_SIZE    = 224
PATCH_SIZE  = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 196
D_MODEL     = 256
N_HEADS     = 8
N_LAYERS    = 4
D_FF        = 512
DROPOUT     = 0.1

ALL_TEMPLATES    = list(range(2, 16))   # [2..15]
NUM_TEMPLATES    = len(ALL_TEMPLATES)   # 14
FREE_TEMPLATES   = {11, 12, 13, 14, 15}
BORDER_TEMPLATES = {2, 3, 4, 5, 6, 7, 8, 9, 10}

LR              = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.01
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5
UPDATE_EPOCHS   = 4
MINI_BATCH_SIZE = 8

DATA_DIR    = "rl_data"
MODEL_PATH  = "rl_data/transformer_rl.pt"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Transformer backbone
# ============================================================

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, D_MODEL,
                              kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerActorCritic(nn.Module):
    """
    Shared Transformer encoder with:
      - template head  : discrete, 14 choices
      - click head     : continuous (x, y) normalised to [0,1]
                         scaled to whichever area window_rect points to
      - value head     : state value V(s)
    """

    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos_embed   = nn.Parameter(torch.zeros(1, NUM_PATCHES+1, D_MODEL))
        self.dropout     = nn.Dropout(DROPOUT)

        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=D_FF, dropout=DROPOUT,
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.norm        = nn.LayerNorm(D_MODEL)

        # Template selection (discrete)
        self.template_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2), nn.GELU(),
            nn.Linear(D_MODEL//2, NUM_TEMPLATES))

        # Click position (continuous) — single head for both free and border
        # Actual area restriction is done by scaling in RLAgent.select_action
        self.click_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2), nn.GELU(),
            nn.Linear(D_MODEL//2, 2),
            nn.Sigmoid())                      # output in [0,1]
        self.click_log_std = nn.Parameter(torch.full((2,), -1.0))

        self.value_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL//2), nn.GELU(),
            nn.Linear(D_MODEL//2, 1))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _encode(self, x):
        B   = x.size(0)
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_embed
        x   = self.dropout(x)
        return self.norm(self.transformer(x))[:, 0]

    def forward(self, x):
        feat = self._encode(x)
        return (self.template_head(feat),
                self.click_head(feat),
                self.click_log_std,
                self.value_head(feat))

    def get_action(self, x):
        t_logits, click_mean, click_log_std, value = self(x)

        t_dist = Categorical(logits=t_logits)
        t_idx  = t_dist.sample()
        t_logp = t_dist.log_prob(t_idx)

        std       = click_log_std.exp().expand_as(click_mean)
        c_dist    = Normal(click_mean, std)
        click_pos = c_dist.sample().clamp(0.0, 1.0)
        c_logp    = c_dist.log_prob(click_pos).sum(dim=-1)

        entropy = t_dist.entropy() + c_dist.entropy().sum(dim=-1)

        action = {
            'template_idx': t_idx.item(),
            'template_num': ALL_TEMPLATES[t_idx.item()],
            'click_norm':   click_pos.squeeze(0).detach(),  # (2,) in [0,1]
        }
        return action, t_logp + c_logp, entropy, value

    def evaluate_actions(self, states, t_actions, c_actions):
        t_logits, click_mean, click_log_std, value = self(states)

        t_dist    = Categorical(logits=t_logits)
        t_logp    = t_dist.log_prob(t_actions)
        t_entropy = t_dist.entropy()

        std    = click_log_std.exp().expand_as(click_mean)
        c_dist = Normal(click_mean, std)
        c_logp = c_dist.log_prob(c_actions).sum(dim=-1)
        c_ent  = c_dist.entropy().sum(dim=-1)

        return t_logp + c_logp, t_entropy + c_ent, value.squeeze(-1)


# ============================================================
# Rollout Buffer
# ============================================================

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states   = []
        self.t_acts   = []
        self.c_acts   = []
        self.logprobs = []
        self.rewards  = []
        self.values   = []
        self.dones    = []

    def add(self, state, t_act, c_act, logprob, reward, value, done):
        self.states.append(state)
        self.t_acts.append(t_act)
        self.c_acts.append(c_act)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self):
        rets, advs, gae, nv = [], [], 0.0, 0.0
        for i in reversed(range(len(self.rewards))):
            mask = 0.0 if self.dones[i] else 1.0
            delta = self.rewards[i] + GAMMA * nv * mask - self.values[i]
            gae   = delta + GAMMA * GAE_LAMBDA * mask * gae
            advs.insert(0, gae)
            rets.insert(0, gae + self.values[i])
            nv = self.values[i]
        adv = torch.tensor(advs, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return torch.tensor(rets, dtype=torch.float32), adv

    def to_tensors(self):
        return (torch.stack(self.states).to(DEVICE),
                torch.tensor(self.t_acts,   dtype=torch.long).to(DEVICE),
                torch.stack(self.c_acts).to(DEVICE),
                torch.tensor(self.logprobs, dtype=torch.float32).to(DEVICE))


# ============================================================
# PPO Trainer
# ============================================================

class PPOTrainer:
    def __init__(self, model):
        self.model     = model
        self.optimizer = optim.Adam(model.parameters(), lr=LR)

    def update(self, buffer):
        returns, advantages = buffer.compute_returns()
        states, t_acts, c_acts, old_logp = buffer.to_tensors()
        returns    = returns.to(DEVICE)
        advantages = advantages.to(DEVICE)
        total_loss = 0.0

        for _ in range(UPDATE_EPOCHS):
            idx = torch.randperm(len(states))
            for s in range(0, len(states), MINI_BATCH_SIZE):
                b = idx[s:s+MINI_BATCH_SIZE]
                if len(b) < 2: continue

                logp, ent, vals = self.model.evaluate_actions(
                    states[b], t_acts[b], c_acts[b])
                ratio  = (logp - old_logp[b]).exp()
                surr   = torch.min(ratio * advantages[b],
                                   ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS) * advantages[b])
                loss   = (-surr.mean()
                          + VALUE_COEF * nn.functional.mse_loss(vals, returns[b])
                          - ENTROPY_COEF * ent.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss


# ============================================================
# RLAgent
# ============================================================

class RLAgent:
    """
    RL agent with strict area enforcement.

    window_rect is set externally by main.py based on template type:
      template 2-10  → wa.as_game_rect()   (left, top, w, h)
      template 11-15 → wa.as_map_rect()    (left, top, w, h)

    The agent's normalised click output (0-1) is scaled within
    whichever rect is currently set, guaranteeing clicks stay inside.
    """

    def __init__(self, window_rect=None, model_path=MODEL_PATH):
        self.model       = TransformerActorCritic().to(DEVICE)
        self.trainer     = PPOTrainer(self.model)
        self.buffer      = RolloutBuffer()
        self.window_rect = window_rect   # (left, top, w, h)
        self.model_path  = model_path
        self.episode     = 0
        self._rewards    = []

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self._last_state = self._last_t = self._last_c = None
        self._last_logp  = self._last_val = None

        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(model_path):
            self.load()
            print(f"[RLAgent] Loaded from {model_path}")
        else:
            print(f"[RLAgent] Fresh model  device={DEVICE}")

    def _pre(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)

    def select_action(self, screenshot, border_corners=None):
        """
        Given a screenshot, output which template to deploy and where to click.

        Click coordinates are scaled within self.window_rect so they ALWAYS
        land inside the area set by main.py (game area or map area).

        Args:
            screenshot     : numpy BGR image
            border_corners : ignored here (used by border_perimeter helper
                             in main.py if needed)

        Returns:
            {
                'template_num': int (2-15),
                'click_x': int,   absolute screen x (within window_rect)
                'click_y': int,   absolute screen y (within window_rect)
            }
        """
        self.model.eval()
        state = self._pre(screenshot)

        with torch.no_grad():
            action, logp, _, value = self.model.get_action(state)

        nx, ny = action['click_norm'][0].item(), action['click_norm'][1].item()

        # Scale normalised (0-1) coords into the currently active window_rect
        if self.window_rect is not None:
            left, top, ww, wh = self.window_rect
            abs_x = int(left + nx * ww)
            abs_y = int(top  + ny * wh)
        else:
            # Fallback: full screen (should not happen in production)
            abs_x = int(nx * 1920)
            abs_y = int(ny * 1080)

        self._last_state = state.squeeze(0).cpu()
        self._last_t     = action['template_idx']
        self._last_c     = action['click_norm'].cpu()
        self._last_logp  = logp.item()
        self._last_val   = value.item()

        print(f"[RLAgent] template={action['template_num']}"
              f"  norm=({nx:.3f},{ny:.3f})"
              f"  screen=({abs_x},{abs_y})"
              f"  rect={self.window_rect}")

        return {
            'template_num': action['template_num'],
            'click_x':      abs_x,
            'click_y':      abs_y,
        }

    def record_reward(self, percentage, done=True):
        reward = float(percentage) if percentage is not None else 0.0
        norm   = (reward - 50.0) / 50.0
        self._rewards.append(reward)
        print(f"[RLAgent] reward={reward:.1f}%  norm={norm:.3f}")

        if self._last_state is not None:
            self.buffer.add(
                state  = self._last_state,
                t_act  = self._last_t,
                c_act  = self._last_c,
                logprob= self._last_logp,
                reward = norm,
                value  = self._last_val,
                done   = done)

    def update(self):
        if not self.buffer.rewards: return
        loss = self.trainer.update(self.buffer)
        self.buffer.clear()
        self.episode += 1
        avg = np.mean(self._rewards[-10:]) if self._rewards else 0
        print(f"[RLAgent] ep={self.episode}  loss={loss:.4f}"
              f"  avg10={avg:.1f}%")
        if self.episode % 10 == 0:
            self.save()
        self._log(loss, avg)

    def _log(self, loss, avg):
        path = os.path.join(DATA_DIR, "training_log.json")
        logs = []
        if os.path.exists(path):
            with open(path) as f: logs = json.load(f)
        logs.append({"ep": self.episode, "loss": round(loss,4),
                     "avg": round(avg,2), "ts": int(time.time())})
        with open(path, "w") as f: json.dump(logs, f, indent=2)

    def save(self):
        torch.save({"model": self.model.state_dict(),
                    "opt":   self.trainer.optimizer.state_dict(),
                    "ep":    self.episode,
                    "rew":   self._rewards}, self.model_path)
        print(f"[RLAgent] Saved → {self.model_path}")

    def load(self):
        ck = torch.load(self.model_path, map_location=DEVICE)
        self.model.load_state_dict(ck["model"])
        self.trainer.optimizer.load_state_dict(ck["opt"])
        self.episode  = ck.get("ep",  0)
        self._rewards = ck.get("rew", [])
        self.model.eval()