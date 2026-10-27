"""Section 15 lab: Vision Transformers (ViT).

Cut an image into patches, embed them, build the ViT input z0, count ViT-B/16
exactly, try CLIP-style zero-shot matching, and interpolate position embeddings
for a higher resolution.

Run:  uv run python 15_Vision_Transformers/lab.py
"""

# %% 1. Setup
import numpy as np

rng = np.random.default_rng(0)
np.set_printoptions(precision=2, suppress=True)


def patchify(img: np.ndarray, P: int) -> np.ndarray:
    """H x W x C image -> N x (P*P*C) patches, row-major, each flattened row by row."""
    H, W, C = img.shape
    assert H % P == 0 and W % P == 0
    x = img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4)
    return x.reshape(-1, P * P * C)


def softmax(z: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# %% 2. Reproduce the worked examples from notes.md
print("6.1 Patch arithmetic")
for H, P in [(224, 16), (224, 8), (224, 32), (384, 16)]:
    N = (H // P) ** 2
    print(f"  {H}x{H}x3, P={P:2d}: N = {N:4d}, {P * P * 3:5d} numbers/patch, "
          f"{N + 1} tokens, attention cost {N**2 / 196**2:.2f}x")

img = np.array([[0, 8, 8, 8], [0, 8, 8, 8], [0, 0, 0, 8], [0, 0, 0, 8]], float)[:, :, None]
E = np.array([[0.25, -0.5], [0.25, 0.5], [0.25, -0.5], [0.25, 0.5]])   # [brightness, right - left]
emb = patchify(img, 2) @ E
print("\n6.2 Patch embeddings [brightness, vertical edge]:\n", emb)
assert np.allclose(emb, [[4, 8], [8, 0], [0, 0], [4, 8]])


def vit_params(d: int, layers: int, d_ff: int, P: int = 16, C: int = 3, img: int = 224, classes: int = 1000) -> int:
    """Patch embedding + [CLS] + positions + encoder blocks (with biases, 2 LN) + final LN + head."""
    N = (img // P) ** 2
    patch = P * P * C * d + d
    block = (4 * d * d + 4 * d) + (2 * d * d_ff + d_ff + d) + 4 * d
    return patch + d + (N + 1) * d + layers * block + 2 * d + d * classes + classes


print(f"\n6.3 ViT-B/16 parameters: {vit_params(768, 12, 3072):,}  (notes: 86,567,656)")
assert vit_params(768, 12, 3072) == 86_567_656

image_emb = np.array([0.6, 0.8])
texts = {"a photo of a cat": [0.8, 0.6], "a photo of a car": [0.0, 1.0], "a photo of a dog": [1.0, 0.0]}
T = np.array(list(texts.values()))
cos = T @ image_emb
print("\n6.4 CLIP zero-shot: cosines", cos, " softmax(x10)", softmax(10 * cos), " softmax(x100)", softmax(100 * cos))
print(f"6.5 MAE: 49 of 196 patches visible -> attention {196**2 / 49**2:.0f}x cheaper")


# %% 3. A general implementation: the ViT input and one encoder layer
def vit_input(img: np.ndarray, P: int, E: np.ndarray, cls: np.ndarray, E_pos: np.ndarray) -> np.ndarray:
    """z0 = [cls; patches @ E] + E_pos, shape (N + 1) x d_model."""
    z = np.vstack([cls[None, :], patchify(img, P) @ E]) + E_pos
    assert z.shape == E_pos.shape
    return z


def encoder_attention(z: np.ndarray) -> np.ndarray:
    """One bidirectional self-attention step with identity projections (no mask)."""
    return softmax(z @ z.T / np.sqrt(z.shape[1])) @ z


P, d = 8, 32
image = rng.random((32, 32, 3))
N = (32 // P) ** 2
z0 = vit_input(image, P, rng.normal(0, 0.05, (P * P * 3, d)), rng.normal(0, 0.05, d), rng.normal(0, 0.05, (N + 1, d)))
print(f"\nViT input for a 32x32x3 image, P = 8: z0 {z0.shape}; after attention {encoder_attention(z0).shape}")


# %% 4. Experiments
def draw_scene() -> np.ndarray:
    """An original 16 x 16 grey-scale drawing: sky, sun, a house with a roof, a tree, grass."""
    s = np.full((16, 16), 6.0)                         # sky
    s[1:4, 12:15] = 9                                  # sun
    s[12:, :] = 3                                      # grass
    s[8:12, 2:8] = 2                                   # house walls
    for r in range(4):                                 # roof
        s[7 - r, 2 + r:8 - r] = 1
    s[10:12, 4:6] = 8                                  # door
    s[6:10, 10:14] = 4                                 # tree crown
    s[10:12, 11:13] = 1                                # trunk
    return s


def experiment_patch_grid() -> None:
    """Patchify a drawing with P = 4 and show each patch's average brightness: the image as 16 tokens."""
    print("\n[Experiment 1] A 16x16 drawing as a 4x4 grid of patch tokens (average brightness)")
    patches = patchify(draw_scene()[:, :, None], 4)
    print(patches.mean(axis=1).reshape(4, 4))
    print("  each of these 16 rows of", patches.shape[1], "numbers is one 'visual token'")


def experiment_conv_equivalence() -> None:
    """Patch embedding == convolution with stride P (computed here with explicit loops)."""
    print("\n[Experiment 2] Patch embedding vs stride-P convolution")
    img3, P, d = rng.random((8, 8, 3)), 4, 5
    Ew = rng.normal(size=(P * P * 3, d))
    filters = Ew.T.reshape(d, P, P, 3)
    conv = np.array([[np.tensordot(img3[r:r + P, c:c + P], filters, axes=([0, 1, 2], [1, 2, 3]))
                      for c in range(0, 8, P)] for r in range(0, 8, P)]).reshape(-1, d)
    print("  identical:", np.allclose(conv, patchify(img3, P) @ Ew))


def interpolate_positions(E_pos: np.ndarray, old: int, new: int) -> np.ndarray:
    """Bilinearly resize a (old*old) x d grid of position embeddings to (new*new) x d."""
    grid = E_pos.reshape(old, old, -1)
    coords = np.linspace(0, old - 1, new)
    i0 = np.floor(coords).astype(int)
    i1 = np.minimum(i0 + 1, old - 1)
    w = coords - i0
    rows = grid[i0] * (1 - w)[:, None, None] + grid[i1] * w[:, None, None]
    out = rows[:, i0] * (1 - w)[None, :, None] + rows[:, i1] * w[None, :, None]
    return out.reshape(new * new, -1)


def experiment_resolution() -> None:
    """Going from 224 to 384 pixels (P = 16): 196 -> 576 patches; positions interpolated 14x14 -> 24x24."""
    print("\n[Experiment 3] Higher resolution")
    E14 = rng.normal(size=(196, 8))
    E24 = interpolate_positions(E14, 14, 24)
    print(f"  positions {E14.shape} -> {E24.shape};  corners kept: "
          f"{np.allclose(E24[0], E14[0]) and np.allclose(E24[-1], E14[-1])}")
    print(f"  attention cost ratio (576/196)^2 = {(576 / 196)**2:.1f}")


experiment_patch_grid()
experiment_conv_equivalence()
experiment_resolution()


# %% 5. Student TODOs
def unpatchify(patches: np.ndarray, P: int, H: int, W: int, C: int) -> np.ndarray:
    """Inverse of patchify: N x (P*P*C) -> H x W x C."""
    # TODO(student): reverse the reshape/transpose steps of patchify.
    raise NotImplementedError


def mae_mask(N: int, ratio: float, rng) -> np.ndarray:
    """Return the sorted indices of the patches that stay VISIBLE (a random 1 - ratio share)."""
    # TODO(student): rng.permutation(N)[: int(N * (1 - ratio))], then sort.
    raise NotImplementedError


def n_tokens(H: int, W: int, P: int, cls: bool = True) -> int:
    """Number of ViT tokens for an H x W image with patch size P."""
    # TODO(student): (H // P) * (W // P), plus one if cls.
    raise NotImplementedError


rimg = rng.random((16, 16, 3))
checks = [
    ("unpatchify", lambda: np.allclose(unpatchify(patchify(rimg, 4), 4, 16, 16, 3), rimg)),
    ("mae_mask", lambda: len(mae_mask(196, 0.75, rng)) == 49 and len(set(mae_mask(196, 0.75, rng))) == 49),
    ("n_tokens", lambda: n_tokens(224, 224, 16) == 197 and n_tokens(336, 336, 14, cls=False) == 576),
]
print("\nStudent TODOs:")
for name, check in checks:
    try:
        print(f"  {name}: {'PASS' if check() else 'FAIL'}")
    except NotImplementedError:
        print(f"  {name}: not implemented yet")
