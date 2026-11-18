"""CNN on log-mel spectrograms (Module 2 — Ajit Kumar Pradhan).

The network is trained with JAX (training-only dependency, see requirements-train.txt) and its
weights are saved as a plain .npz file. Inference uses the pure-NumPy forward pass below, so
the app itself needs no deep-learning framework.

Architecture (input 1 × 64 mel bands × T frames, 64 ms per frame):
  3 × [3×3 conv → ReLU → 2×2 max-pool]   (16, 32, 48 channels)   → local spectral patterns
  average over frequency                                          → 48 × T/8 sequence
  2 × [dilated temporal conv, k=9, dilation 2 and 4 → ReLU]       → ~20 s receptive field,
                                                                    long enough to "see" a
                                                                    10+ s breathing pause
  global max + mean pooling over time → dense 32 → dense 1 → sigmoid
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CONV2D = [(1, 16), (16, 32), (32, 48)]
CONV1D = [(48, 64, 2), (64, 64, 4)]
KERNEL1D = 9


def init_params(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    p = {}
    for i, (cin, cout) in enumerate(CONV2D):
        p[f"c{i}_w"] = (rng.standard_normal((cout, cin, 3, 3)) * np.sqrt(2 / (cin * 9))).astype(np.float32)
        p[f"c{i}_b"] = np.zeros(cout, np.float32)
    for i, (cin, cout, _) in enumerate(CONV1D):
        p[f"t{i}_w"] = (rng.standard_normal((cout, cin, KERNEL1D)) * np.sqrt(2 / (cin * KERNEL1D))).astype(np.float32)
        p[f"t{i}_b"] = np.zeros(cout, np.float32)
    p["d0_w"] = (rng.standard_normal((128, 32)) * np.sqrt(2 / 128)).astype(np.float32)
    p["d0_b"] = np.zeros(32, np.float32)
    p["d1_w"] = (rng.standard_normal((32, 1)) * np.sqrt(1 / 32)).astype(np.float32)
    p["d1_b"] = np.zeros(1, np.float32)
    return p


# --------------------------------------------------------------------------- NumPy inference
def _conv2d_same(x, w, b):            # x (C,H,W), w (O,C,3,3)
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    win = sliding_window_view(xp, (3, 3), axis=(1, 2))          # (C,H,W,3,3)
    return np.einsum("chwij,ocij->ohw", win, w, optimize=True) + b[:, None, None]


def _pool2(x):                        # (C,H,W) → (C,H/2,W/2), floor
    c, h, w = x.shape
    x = x[:, : h // 2 * 2, : w // 2 * 2]
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def _conv1d_same(x, w, b, d):         # x (C,T), w (O,C,K)
    k = w.shape[2]
    pad = (k - 1) * d // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    idx = np.arange(x.shape[1])[:, None] + np.arange(k)[None, :] * d   # (T,K)
    win = xp[:, idx]                                                   # (C,T,K)
    return np.einsum("ctk,ock->ot", win, w, optimize=True) + b[:, None]


def predict_proba(params: dict, mel: np.ndarray) -> float:
    """Probability of an irregular pattern for one log-mel window (n_mels × T)."""
    x = mel[None].astype(np.float32)
    for i in range(len(CONV2D)):
        x = _pool2(np.maximum(_conv2d_same(x, params[f"c{i}_w"], params[f"c{i}_b"]), 0))
    x = x.mean(axis=1)                                                  # (48, T/8)
    for i, (_, _, d) in enumerate(CONV1D):
        x = np.maximum(_conv1d_same(x, params[f"t{i}_w"], params[f"t{i}_b"], d), 0)
    h = np.concatenate([x.max(axis=1), x.mean(axis=1)])
    h = np.maximum(h @ params["d0_w"] + params["d0_b"], 0)
    z = float((h @ params["d1_w"] + params["d1_b"])[0])
    return float(1 / (1 + np.exp(-z)))


def load(path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# --------------------------------------------------------------------------- JAX training
def train(X_tr, y_tr, X_va, y_va, epochs: int = 25, batch: int = 32, lr: float = 1e-3, seed: int = 0,
          log=print) -> tuple[dict, list[dict]]:
    """Train with JAX/Optax. Returns (best params as NumPy arrays, history)."""
    import jax
    import jax.numpy as jnp
    import optax

    def conv2d(x, w, b):
        y = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW"))
        return y + b[None, :, None, None]

    def pool2(x):
        n, c, h, w = x.shape
        x = x[:, :, : h // 2 * 2, : w // 2 * 2]
        return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))

    def conv1d(x, w, b, d):
        k = w.shape[2]
        pad = (k - 1) * d // 2
        y = jax.lax.conv_general_dilated(x, w, (1,), [(pad, pad)], rhs_dilation=(d,),
                                         dimension_numbers=("NCH", "OIH", "NCH"))
        return y + b[None, :, None]

    def forward(p, x):                      # x (N,1,64,T) → logits (N,)
        for i in range(len(CONV2D)):
            x = pool2(jax.nn.relu(conv2d(x, p[f"c{i}_w"], p[f"c{i}_b"])))
        x = x.mean(axis=2)
        for i, (_, _, d) in enumerate(CONV1D):
            x = jax.nn.relu(conv1d(x, p[f"t{i}_w"], p[f"t{i}_b"], d))
        h = jnp.concatenate([x.max(axis=2), x.mean(axis=2)], axis=1)
        h = jax.nn.relu(h @ p["d0_w"] + p["d0_b"])
        return (h @ p["d1_w"] + p["d1_b"])[:, 0]

    def loss_fn(p, x, y):
        logits = forward(p, x)
        return optax.sigmoid_binary_cross_entropy(logits, y).mean()

    params = {k: jnp.asarray(v) for k, v in init_params(seed).items()}
    opt = optax.adamw(optax.cosine_decay_schedule(lr, epochs * (len(X_tr) // batch + 1)), weight_decay=1e-4)
    state = opt.init(params)

    @jax.jit
    def step(p, s, x, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, y)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, l

    predict = jax.jit(lambda p, x: jax.nn.sigmoid(forward(p, x)))
    rng = np.random.default_rng(seed)
    best, best_acc, history = None, -1.0, []
    xv = jnp.asarray(X_va[:, None])
    for ep in range(epochs):
        order = rng.permutation(len(X_tr))
        losses = []
        for i in range(0, len(order) - batch + 1, batch):
            idx = order[i:i + batch]
            xb = X_tr[idx].copy()
            # augmentation: random circular time shift + one frequency mask (SpecAugment-style)
            for j in range(len(xb)):
                xb[j] = np.roll(xb[j], rng.integers(-30, 30), axis=1)
                f0 = rng.integers(0, xb.shape[1] - 8)
                xb[j, f0:f0 + rng.integers(0, 8)] = -1.0
            params, state, l = step(params, state, jnp.asarray(xb[:, None]), jnp.asarray(y_tr[idx], jnp.float32))
            losses.append(float(l))
        pv = np.concatenate([np.asarray(predict(params, xv[i:i + 64])) for i in range(0, len(xv), 64)])
        acc = float(((pv >= 0.5) == y_va).mean())
        history.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_acc": acc})
        log(f"  epoch {ep + 1:2d}  loss {np.mean(losses):.4f}  val acc {acc:.3f}")
        if acc >= best_acc:
            best_acc, best = acc, {k: np.asarray(v) for k, v in params.items()}
    # the app uses the NumPy forward pass — check it matches JAX exactly
    pj = np.asarray(predict({k: jnp.asarray(v) for k, v in best.items()}, xv[:8]))
    pn = np.array([predict_proba(best, m) for m in X_va[:8]])
    log(f"  NumPy vs JAX forward pass: max |diff| = {np.abs(pj - pn).max():.2e}")
    return best, history
