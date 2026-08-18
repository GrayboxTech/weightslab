# wl-video-generation — video (and image) generation under WeightsLab supervision, with sound

Train a small rectified-flow video generator from scratch, on a laptop, and
watch it **per clip** in the studio: poster frames in the grid, a real player
with frame scrubbing and audio in the modal.

## Why from scratch instead of a LoRA on a real video model?

Because the point is the *supervision loop*, not the sample quality. LTX-Video
or CogVideoX need ~24 GB of VRAM to fine-tune; this model trains on a 4 GB
laptop GPU (and on CPU), so the whole loop — spot a bad clip, tag it, remove
it, watch the loss trajectory change — is something you can actually run. The
integration it exercises is identical to what a 2B-parameter model would use.

## The three input modes

One `mode` knob in `config.yaml` selects what conditions the generation:

| `mode`       | Conditioning            | What it demonstrates            |
|--------------|-------------------------|---------------------------------|
| `text`       | prompt                  | text-to-video                   |
| `text_video` | prompt + source clip    | instructed video editing        |
| `video`      | source clip             | a learned "anime" style filter  |

Set **`frames: 1`** and the same script is an **image** generator — a clip of
length one is just a picture, and every path is length-agnostic.

```
   text        "a red ball bouncing" ─────────────────┐
                                                      ├──▶ 3D U-Net ──▶ clip + audio
   text_video  prompt + source clip ──────────────────┤     (flow
                                                      │    matching)
   video       source clip ───────────────────────────┘
```

## Pipeline

```
utils/data.py    clips + captions + audio  ->  (target, source), idx, caption, metadata
                 └─ task_type = "video_generation"   -> studio routes to the video path
                 └─ get_audio(index)                 -> soundtrack the player muxes in
                 └─ fps / num_frames                 -> grid badge, no decoding needed

utils/model.py   VideoFlowUNet   [B,3,T,H,W] + t (+ text) (+ source) -> velocity
                 FlowMatchingLoss(reduction="none")  -> [B], one loss per clip

main.py          rectified-flow loop, wl.watch_or_edit on every object
```

The objective is rectified flow (the same family as FLUX / SD3):

```
    x_t    = (1 - t) · clip + t · noise
    target = noise - clip                 (the velocity field)
    loss_i = mean( (v_pred_i - target_i)^2 )      -> one scalar per clip
```

## Supervision: what to actually watch in the studio

| Signal | Reading | Action |
|---|---|---|
| `train/fm_loss` flat and high on a clip | bad caption, or a clip that does not match its prompt | tag it, then remove it from the loader live |
| `train/fm_loss` collapsing to ~0 on a clip | the model is memorizing that clip | down-weight it or add augmentation |
| `test/fm_loss` rising while train falls | overfitting the small clip set | raise `max_samples`, or stop |
| Loss-shape tag `plateaued` | that clip stopped contributing | candidate for removal |

Because `FlowMatchingLoss` is wrapped with `flag="loss"` and called with
`batch_ids=uids`, every clip gets its own loss trajectory, and the signal is
auto-enrolled in the loss-shape classifier. Right-click the signal in the left
panel → **Plot signal trajectory** to draw the per-clip curves.

## In the UI

* **Grid / list** — one **poster frame** per clip, badged `▶ 2.0s ♪`. Clip bytes
  are never sent while browsing, so a video dataset browses as fast as an image
  one.
* **Modal** — press **Play**: the muxed H.264 + AAC clip streams over the
  `GetMedia` RPC into the player. Scrub with the frame strip (`‹ ▮▮▮ ›`,
  `frame 7 / 16`), hear the soundtrack, and hit the expand button for a
  near-fullscreen view.

## Datasets (all public)

`data.source` in `config.yaml`:

| value | what | size | captions | audio |
|---|---|---|---|---|
| `synthetic` *(default)* | procedural shapes, **no download** | 0 | generated | **synthesized, phase-locked to motion** |
| `disney` | [`sayakpaul/video-dataset-disney-organized`](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) | 24 MB | real, long, properly joined | synthesized (source is silent) |
| `kinetics` | [`nateraw/kinetics-mini`](https://huggingface.co/datasets/nateraw/kinetics-mini) | 136 MB | templated from class | **real AAC** |

Notes on the choices:

* **WebVid is dead** — Shutterstock C&D'd it in Feb 2024 and the URLs no longer
  resolve. Any tutorial pointing at it will fail today.
* Clips are fetched with `huggingface_hub` and decoded with **OpenCV**,
  deliberately *not* through `datasets` — its v4 `Video` feature requires
  `torchcodec`, an FFmpeg- and torch-version-matched build that routinely fails
  on Windows.
* `synthetic` is the default and is the better teaching choice for the audio
  half: real Kinetics audio is ambient YouTube noise with near-zero correlation
  to the visuals, so a small model cannot learn A/V alignment from it. The
  synthetic track is derived from the clip's own motion, so the coupling is
  real and learnable.

## Run

```bash
# 0. deps — torch/numpy/opencv come with weightslab; nothing else is required.
#    ffmpeg gives you MP4 + audio; without it the studio falls back to silent GIF.
pip install imageio-ffmpeg          # or have ffmpeg on PATH

# 1. train (starts the gRPC service in-process)
python main.py

# 2. open the studio
#    grid/list -> poster frames;  double-click a clip -> "Play"
```

Switch modes by editing `config.yaml`:

```yaml
mode: text          # text | text_video | video
frames: 16          # 1 => image generation
resolution: 64
data:
  source: synthetic # synthetic | disney | kinetics
```

## Hardware

Defaults (16 frames @ 64×64, `base_channels: 48`, ~5 M params, batch 2) fit in
well under 4 GB of VRAM and also run on CPU. Raise `base_channels` and
`resolution` once you have a real GPU.

## Gotchas worth knowing

* `wl.watch_or_edit(cfg, flag="hyperparameters")` replaces the scalars in `cfg`
  with live `ValueProxy` objects. Anything feeding a tensor shape, a `numpy`
  call or a `range()` **must** be coerced with `int()` / `float()` first — a
  proxy is not a number. `main.py` does this right after the call.
* `FlowMatchingLoss` takes `target=` as a **keyword**. The watched-loss wrapper
  reads the 2nd *positional* argument as per-sample "targets" for logging,
  which is right for a class vector and wrong for a full velocity tensor.
* `__getitem__` returns `inputs` as a **tuple** whose first element is the clip
  to preview. WeightsLab plucks element `[0]` of a nested tuple for the poster
  frame; a dict there breaks preview extraction entirely.
