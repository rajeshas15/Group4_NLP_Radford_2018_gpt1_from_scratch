
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT1Config:
    vocab_size: int = 40_478           # paper: BPE vocab 40k
    n_ctx: int = 512                   # paper: context window 512
    n_layer: int = 12                  # paper: 12 blocks
    n_head: int = 12                   # paper: 12 heads
    d_model: int = 768                 # paper: 768 hidden
    d_ff: int = 3072                   # paper: 4 * d_model
    dropout: float = 0.1               # paper: 0.1 everywhere
    pad_id: int = 0


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation as used in the paper (tanh approximation, Hendrycks & Gimpel 2016)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention. Eq. (1) of "Attention Is All You Need",
    with a lower-triangular mask so position i can only attend to positions <= i.
    """

    def __init__(self, cfg: GPT1Config):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        # Fused QKV projection -- cheaper than three separate Linears.
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Persistent causal mask up to max context length.
        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                        # (B, n_head, T, d_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # concat heads
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise feed-forward sub-layer: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, cfg: GPT1Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(gelu(self.fc1(x))))


class Block(nn.Module):
    """One transformer block. Paper uses post-LayerNorm (LN after residual add)."""

    def __init__(self, cfg: GPT1Config):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x


class GPT1(nn.Module):
    """Decoder-only Transformer language model (GPT-1)."""

    def __init__(self, cfg: GPT1Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.n_ctx, cfg.d_model)     # learned, per paper
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Tie the LM head weights to the input embedding (P(u) = softmax(h W_e^T)).
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Paper: N(0, 0.02) init.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,                # (B, T) int64
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.n_ctx, f"sequence length {T} exceeds n_ctx {self.cfg.n_ctx}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)
        h = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)                                # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Greedy/temperature sampling for quick qualitative checks."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.n_ctx:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
