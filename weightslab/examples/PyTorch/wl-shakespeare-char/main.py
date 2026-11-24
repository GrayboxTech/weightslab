"""Character-level GPT on Karpathy's tiny Shakespeare, under WeightsLab.

A GPT from the baseline zoo (NanoGPT by default, MicroGPT with
``model.class: MicroGPT``) learns to predict the next character of the
complete works' 1.1 MB excerpt that char-rnn and nanoGPT made famous. The
text is downloaded on the first run.

Each sample is a window of ``block_size`` characters: the input ids, and the
same window shifted by one as the target. The first 90% of the text gives
the training windows, the last 10% the test windows. In Weights Studio:

- the grid shows every sample as its text (``get_items``), sortable by its
  loss and its share of next characters predicted right;
- the Modeling tab shows the blocks, the attention of each head, and the
  tokens as characters (the model carries its alphabet as ``itos``);
- "Why do these fail?" works on any selection of windows.

Every ``sample_every`` steps the model writes a short sample of "Shakespeare"
to the console and to ``<root_log_dir>/samples.txt``.

Run:  python main.py   (then open the URL it prints, and press Train)
"""
import itertools
import os
import ssl
import tempfile
import time
import urllib.request

# Windows SSL fix: some Windows cert stores contain malformed ASN1 certs that
# crash ssl.create_default_context(). Fall back to unverified only when broken.
try:
    ssl.create_default_context()
except ssl.SSLError:
    ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml
from torch.utils.data import Dataset

import weightslab as wl
from weightslab.examples.utils.baseline_models.pytorch import models as zoo

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


# =============================================================================
# Data
# =============================================================================
def load_text(data_root: str) -> str:
    """tiny Shakespeare, downloaded once into *data_root*."""
    os.makedirs(data_root, exist_ok=True)
    path = os.path.join(data_root, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print(f"Downloading tiny Shakespeare to {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class CharWindows(Dataset):
    """Windows of ``block`` characters of *text*, one every ``stride``:
    (input ids, the next character's id at each position)."""

    def __init__(self, text: str, stoi: dict, itos: list, block: int, stride: int, offset: int = 0):
        self.ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        self.itos = itos
        self.block = block
        self.offset = offset            # where *text* starts in the whole corpus
        self.starts = list(range(0, len(self.ids) - block - 1, max(1, stride)))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        p = self.starts[i]
        return self.ids[p:p + self.block], self.ids[p + 1:p + self.block + 1]

    def get_items(self, i, include_metadata=False, include_labels=False, include_images=False):
        """What the Studio shows: the window as text, and where it sits in the corpus."""
        x, y = self[i]
        text = "".join(self.itos[int(t)] for t in x) if include_images else None
        meta = {"char_offset": self.offset + self.starts[i]} if include_metadata else {}
        return text, str(i), (y.numpy() if include_labels else None), meta


# =============================================================================
# Loss, train / test steps, sampling
# =============================================================================
class SequenceCE(nn.Module):
    """Cross-entropy per sample: the mean over its positions (reduction="none"
    per sample, which is what WeightsLab records and the grid sorts on)."""

    def forward(self, logits, targets):
        return F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").mean(dim=1)


def train(loader, model, optimizer, criterion, device):
    with wl.guard_training_context:
        inputs, ids, targets = next(loader)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.float(), targets.long(), batch_ids=ids, preds=logits.argmax(-1)).mean()
        loss.backward()
        optimizer.step()
    return loss.detach().cpu().item()


def test(loader, model, criterion, device):
    """Full pass over the test windows: mean loss and next-character accuracy."""
    total, right, count, n = 0.0, 0, 0, 0
    for inputs, ids, targets in loader:
        with wl.guard_testing_context:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(-1)
            loss = criterion(logits.float(), targets.long(), batch_ids=ids, preds=preds)
            acc = (preds == targets).float().mean(dim=1)          # per window
            wl.save_signals(batch_ids=ids, signals={"test_metric/char_accuracy": acc},
                            preds_raw=logits, targets=targets, preds=preds)
            total += float(loss.sum())
            right += int((preds == targets).sum())
            count += int(targets.numel())
            n += int(loss.numel())
    return total / max(n, 1), 100.0 * right / max(count, 1)


@torch.no_grad()
def sample(model, stoi, itos, block, device, length=300, prompt="\n", temperature=0.8):
    """Generate *length* characters after *prompt*, one at a time."""
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    for _ in range(length):
        logits = model(idx[:, -block:])[:, -1, :] / temperature
        nxt = torch.multinomial(F.softmax(logits.float(), dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return "".join(itos[int(t)] for t in idx[0, len(prompt):])


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()

    parameters = {}
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    parameters.setdefault("experiment_name", "shakespeare_char")
    parameters.setdefault("device", "auto")
    wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]

    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = tempfile.mkdtemp()
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    log_dir = parameters["root_log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    # ---- data ----
    data_cfg = parameters.get("data", {})
    text = load_text(parameters.get("data_root") or os.path.join(log_dir, "data"))
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    model_cfg = parameters.get("model", {})
    block = int(model_cfg.get("block_size", 64))
    cut = int(len(text) * float(data_cfg.get("train_share", 0.9)))
    train_cfg = data_cfg.get("train_loader", {})
    test_cfg = data_cfg.get("test_loader", {})
    _train_dataset = CharWindows(text[:cut], stoi, chars, block,
                                 stride=int(train_cfg.get("stride", block // 2)))
    _test_dataset = CharWindows(text[cut:], stoi, chars, block,
                                stride=int(test_cfg.get("stride", block)), offset=cut)

    # ---- model ----
    cls = getattr(zoo, model_cfg.get("class", "NanoGPT"))
    kwargs = {k: int(model_cfg[k]) for k in ("n_embd", "n_head", "n_layer") if k in model_cfg}
    if cls is zoo.NanoGPT and "dropout" in model_cfg:
        kwargs["dropout"] = float(model_cfg["dropout"])
    _model = cls(vocab_size=len(chars), block_size=block, **kwargs)
    _model.itos = chars                 # the Modeling tab shows tokens as characters
    model = wl.watch_or_edit(_model, flag="model", device=device,
                             compute_dependencies=True,
                             skip_previous_auto_load=parameters.get("skip_checkpoint_load", False))
    optimizer = wl.watch_or_edit(
        torch.optim.AdamW(model.parameters(), lr=float(parameters.get("optimizer", {}).get("lr", 1e-3))),
        flag="optimizer")

    enable_h5 = parameters.get("ledger_enable_h5_persistence", True)
    train_loader = wl.watch_or_edit(
        _train_dataset, flag="data", loader_name="train_loader",
        batch_size=int(train_cfg.get("batch_size", 64)), shuffle=train_cfg.get("shuffle", True),
        is_training=True, compute_hash=False, preload_labels=False, preload_metadata=False,
        enable_h5_persistence=enable_h5)
    test_loader = wl.watch_or_edit(
        _test_dataset, flag="data", loader_name="test_loader",
        batch_size=int(test_cfg.get("batch_size", 128)), shuffle=False,
        is_training=False, compute_hash=False, preload_labels=False, preload_metadata=False,
        enable_h5_persistence=enable_h5)

    train_criterion = wl.watch_or_edit(SequenceCE(), flag="loss", signal_name="train-loss-CE", log=True)
    test_criterion = wl.watch_or_edit(SequenceCE(), flag="loss", signal_name="test-loss-CE", log=True)

    wl.serve(serving_grpc=parameters.get("serving_grpc", True))

    eval_every = int(parameters.get("eval_full_to_train_steps_ratio", 500))
    sample_every = int(parameters.get("sample_every", 500))
    samples_path = os.path.join(log_dir, "samples.txt")
    n_params = sum(p.numel() for p in _model.parameters())
    print("=" * 60)
    print(f" tiny Shakespeare: {len(text):,} characters, {len(chars)} distinct")
    print(f" {type(_model).__name__}: {n_params / 1e6:.2f}M parameters, block {block}, on {device}")
    print(f" windows: train={len(_train_dataset):,}, test={len(_test_dataset):,}")
    print(f" test every {eval_every} steps, sample every {sample_every}; logs in {log_dir}")
    print(" Press Train in Weights Studio to start.")
    print("=" * 60 + "\n")

    # Training runs until you stop it (Pause in the Studio, or Ctrl+C):
    # inspect, edit the data or the model, keep going.
    tqdm_display = parameters.get("tqdm_display", True)
    steps = tqdm.tqdm(itertools.count(), desc="Training", ncols=120,
                      bar_format="{desc}: {n} steps [{elapsed}, {rate_fmt}] | {postfix}") \
        if tqdm_display else itertools.count()
    test_loss = test_acc = None
    for _ in steps:
        train_loss = train(train_loader, model, optimizer, train_criterion, device)
        age = model.get_age() if hasattr(model, "get_age") else 0
        if age > 0 and age % eval_every == 0:
            test_loss, test_acc = test(test_loader, model, test_criterion, device)
        if age > 0 and age % sample_every == 0:
            text_out = sample(model, stoi, chars, block, device)
            with open(samples_path, "a", encoding="utf-8") as fh:
                fh.write(f"===== step {age} =====\n{text_out}\n\n")
            (steps.write if tqdm_display else print)(f"\n----- sample at step {age} -----\n{text_out}\n")
        if tqdm_display:
            post = f"train_loss={train_loss:.3f}"
            if test_loss is not None:
                post += f" | test_loss={test_loss:.3f} | test_char_acc={test_acc:.1f}%"
            steps.set_postfix_str(post)

    print(f"\nTraining stopped after {time.time() - start_time:.0f}s; logs in {log_dir}")
    wl.write_history()
    wl.write_dataframe()
    wl.keep_serving()
