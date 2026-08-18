"""A small latent-free video diffusion model, trained from scratch.

Deliberately compact (~5-15 M params depending on config) so the example runs
on a 4 GB laptop GPU and on CPU. It is a 3D U-Net that predicts a rectified-flow
velocity field over a clip, conditioned in one of three ways:

  * text        a bag-of-hashed-words prompt embedding
  * text_video  the same prompt embedding + the source clip concatenated
                channel-wise (instructed editing)
  * video       the source clip only (a learned style filter)

The text encoder is a hashing embedding rather than a pretrained transformer on
purpose: it needs no download, no tokenizer, and no extra dependency, which
keeps the example runnable offline. It is enough to distinguish the prompts in
these datasets; it is not a substitute for a real text encoder at scale.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


TEXT_VOCAB_BUCKETS = 4096


def encode_captions(captions, dim, device):
    """Hash captions into a fixed-size bag-of-words vector in [0, 1].

    Deterministic and dependency-free. Returns ``[B, TEXT_VOCAB_BUCKETS]``
    which the model projects down to ``dim``.
    """
    batch = torch.zeros(len(captions), TEXT_VOCAB_BUCKETS, device=device)
    for i, caption in enumerate(captions):
        words = str(caption).lower().replace(",", " ").replace(".", " ").split()
        if not words:
            continue
        for word in words:
            batch[i, hash(word) % TEXT_VOCAB_BUCKETS] += 1.0
        batch[i] /= max(1.0, float(len(words)))
    return batch


def timestep_embedding(t, dim):
    """Standard sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock3D(nn.Module):
    """3D residual block with FiLM-style conditioning from the embedding."""

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(emb)[:, :, None, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class VideoFlowUNet(nn.Module):
    """3D U-Net predicting the rectified-flow velocity for a clip.

    Input  ``[B, C_in, T, H, W]``  (C_in = 3, or 6 when a source clip conditions)
    Output ``[B, 3, T, H, W]``
    """

    def __init__(self, mode="text", base_channels=48, channel_mults=(1, 2, 4),
                 emb_dim=192):
        super().__init__()
        self.mode = mode
        self.uses_text = mode in ("text", "text_video")
        self.uses_source = mode in ("text_video", "video")

        in_channels = 6 if self.uses_source else 3

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.text_proj = (
            nn.Sequential(nn.Linear(TEXT_VOCAB_BUCKETS, emb_dim), nn.SiLU(),
                          nn.Linear(emb_dim, emb_dim))
            if self.uses_text else None)

        self.stem = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_mults]
        self.downs = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        prev = base_channels
        for ch in channels:
            self.downs.append(ResBlock3D(prev, ch, emb_dim))
            # Downsample spatially only — clips here are short, and halving T
            # repeatedly would destroy the motion the model must learn.
            self.downsamplers.append(nn.Conv3d(ch, ch, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
            prev = ch

        self.mid = ResBlock3D(prev, prev, emb_dim)

        self.ups = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for ch in reversed(channels):
            self.upsamplers.append(
                nn.ConvTranspose3d(prev, ch, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)))
            # Skip connection doubles the channel count entering the block.
            self.ups.append(ResBlock3D(ch * 2, ch, emb_dim))
            prev = ch

        self.out_norm = nn.GroupNorm(min(8, prev), prev)
        self.out_conv = nn.Conv3d(prev, 3, 3, padding=1)
        self.emb_dim = emb_dim

    def forward(self, x, t, captions=None, source=None):
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim))
        if self.uses_text and self.text_proj is not None:
            if captions is None:
                captions = [""] * x.shape[0]
            emb = emb + self.text_proj(encode_captions(captions, self.emb_dim, x.device))

        if self.uses_source:
            if source is None:
                source = torch.zeros_like(x)
            x = torch.cat([x, source], dim=1)

        h = self.stem(x)
        skips = []
        for block, down in zip(self.downs, self.downsamplers):
            h = block(h, emb)
            skips.append(h)
            h = down(h)

        h = self.mid(h, emb)

        for block, up in zip(self.ups, self.upsamplers):
            h = up(h)
            skip = skips.pop()
            # Transposed conv can be off by a pixel on odd sizes.
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-3:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), emb)

        return self.out_conv(F.silu(self.out_norm(h)))


class FlowMatchingLoss(nn.Module):
    """Rectified-flow objective, one scalar per sample.

        loss_i = mean_over_dims( (v_pred_i - (noise_i - x1_i))^2 )

    Returns shape ``[B]`` (reduction="none"), which is what
    ``wl.watch_or_edit(flag="loss")`` expects: the wrapper logs this per-sample
    tensor under ``signal_name`` when you pass ``batch_ids``.

    IMPORTANT: pass ``target`` as a KEYWORD. The wl wrapper reads the 2nd
    POSITIONAL argument as per-sample "targets" for logging — right for a class
    vector, wrong for a full velocity tensor.
    """

    def forward(self, model_pred, target):
        err = (model_pred.float() - target.float()) ** 2
        return err.reshape(err.shape[0], -1).mean(dim=1)


@torch.no_grad()
def sample_clip(model, shape, device, steps=25, captions=None, source=None):
    """Euler-integrate the flow from noise back to a clip, in [-1, 1]."""
    x = torch.randn(shape, device=device)
    for i in range(steps):
        t_val = 1.0 - i / steps
        t = torch.full((shape[0],), t_val, device=device)
        v = model(x, t * 1000.0, captions=captions, source=source)
        x = x - v * (1.0 / steps)
    return x.clamp(-1, 1)
