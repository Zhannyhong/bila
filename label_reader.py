"""
Reading call-number stickers from a camera image: find each sticker, read
its two lines (the Dewey class, e.g. "031.0", over the author mark, "PDP"),
and match what was read to the catalogue, or say it can't be read.

    from label_reader import LabelReader
    reader = LabelReader("~/bila_data/labels/finder.pt", "~/bila_data/labels/reader.pt", catalogue=texts)
    for r in reader.read(rgb):        # Reading: corners (px), text (a catalogue entry or None), confidence
        ...

    # in WSL, on make_vision_dataset.py's images (needs `uv sync --group rl`)
    uv run python label_reader.py train-finder --data ~/bila_data/train --out ~/bila_data/labels
    uv run python label_reader.py train-reader --data ~/bila_data/train --val ~/bila_data/val --out ~/bila_data/labels
    uv run python label_reader.py eval --data ~/bila_data/val --weights ~/bila_data/labels

Three steps, each a plain function of the image:
  1. Find (StickerFinder): a small segmentation network (torchvision's
     LR-ASPP MobileNetV3, fine-tuned from COCO) marks each sticker's core,
     the middle 80% of it, so that stickers on neighbouring books, which can
     touch, come out as separate pieces. Each piece's outline, as a
     quadrilateral, grown back to the whole sticker, gives its 4 corners. (A
     colour rule doesn't do: white covers and printed labels merge with the
     stickers, and small stickers fall apart into pieces.)
  2. Read (Reader): the sticker is rectified from its corners and its two
     lines cut out at their fixed places on the label; a small CRNN (conv +
     LSTM, CTC over characters) reads each. The camera may see the sticker
     turned (a leaning book, the tray from above), so it is read in all four
     orientations.
  3. Match: each catalogue entry is scored by the network's probability of
     reading it (the CTC likelihood, both lines, best orientation); the
     confidence is the best entry's share of all entries' probability. Less
     than MIN_CONFIDENCE, or less likely per character than MIN_CHAR_LOGP:
     unreadable (text None). Without a catalogue, the network's own best
     reading and its probability.

Reading close up works; from the cart's fixed cameras the stickers are small
(row_cam: ~45 px tall, tray_cam: ~25 px), and they mostly serve to find the
stickers for a closer look (see `eval` for the numbers per camera and
distance).
"""

from __future__ import annotations

# Rendering (MuJoCo's EGL) after torch is imported crashes unless this is
# imported first; whatever imports this module may render.
import torch._dynamo  # noqa: F401  (see above)

import argparse
import collections
import glob
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CORE = 0.6                      # the finder marks this middle share of each sticker (each side), not its edges:
                                # at 0.8, neighbours' cores ran together in row_cam and tray_cam images
FINDER_WEIGHT = 10.0            # weight of sticker pixels in the finder's loss (they are ~1% of an image)
MIN_SEEN = 0.9                  # a sticker less visible than this is neither a target nor a mistake for the finder
MIN_CORE_PX = 12                # a marked piece smaller than this isn't a sticker
CANON_H = 96                    # px: rectified sticker height
LINES = ((0.10, 0.40), (0.33, 0.62))   # rows of the rectified sticker each line is cut from (cart_demo.sticker)
STRIP = (32, 128)               # (height, width) each line is read at
CHARSET = "0123456789.-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MIN_CONFIDENCE = 0.99           # catalogue match: below this share of the probability, unreadable (val:
                                # 0.15% of found stickers read wrong at 0.99, 0.8% at 0.9; a misread misplaces a book)
MIN_CHAR_LOGP = -1.0            # catalogue match: the best entry less likely than this per character: unreadable
JITTER = 0.04                   # reader training: corners moved by up to this share of the sticker's size
IMAGENET = (np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32))
DISTANCE_BINS = (0.0, 0.25, 0.5, 1.0, 9.0)   # m, for the evaluation


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def scaled(quad: np.ndarray, s: float) -> np.ndarray:
    """The quadrilateral scaled by s about its centre."""
    c = quad.mean(axis=0)
    return c + (quad - c) * s


def clockwise(quad: np.ndarray) -> np.ndarray:
    """The 4 corners in clockwise order as seen in the image (y down), as
    reading order is: top-left, top-right, bottom-right, bottom-left."""
    x, y = quad[:, 0], quad[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return quad if area > 0 else quad[::-1].copy()


def quads_from_core(core: np.ndarray, min_px: int = MIN_CORE_PX) -> List[np.ndarray]:
    """Sticker corners (4, 2), clockwise, from a (H, W) bool map of sticker
    cores: each connected piece's outline as a quadrilateral (or, if it won't
    simplify to one, its smallest rectangle), grown from the core to the
    whole sticker."""
    n, pieces, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        contours, _ = cv2.findContours((pieces == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea)).astype(np.float32)
        quad = None
        for eps in (0.02, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2)
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(hull))
        # Pixel centres: the piece reaches half a pixel past its outermost ones.
        quad = scaled(quad.astype(np.float64), 1 + 1.0 / max(np.ptp(quad, axis=0).min(), 1.0))
        out.append(clockwise(scaled(quad, 1 / CORE)))
    return out


def rectified(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """The sticker straightened from its corners (top-left, top-right,
    bottom-right, bottom-left of its text): CANON_H high, as wide as its own
    aspect."""
    tl, tr, br, bl = np.asarray(corners, np.float32)
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    W = int(np.clip(round(CANON_H * w / max(h, 1e-6)), 16, 4 * CANON_H))
    P = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]),
                                    np.float32([[0, 0], [W, 0], [W, CANON_H], [0, CANON_H]]))
    return cv2.warpPerspective(gray, P, (W, CANON_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def line_strips(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """(2, *STRIP) uint8: the sticker's two lines, cut from where
    cart_demo.sticker prints them and resized for the reader."""
    sticker = rectified(gray, corners)
    out = []
    for top, bottom in LINES:
        line = sticker[int(top * CANON_H):int(bottom * CANON_H)]
        out.append(cv2.resize(line, STRIP[::-1], interpolation=cv2.INTER_AREA if line.shape[1] > STRIP[1]
                              else cv2.INTER_LINEAR))
    return np.stack(out)


def turned(corners: np.ndarray, k: int) -> np.ndarray:
    """The corners as if the sticker were turned k quarter turns: reading
    order starting k corners on."""
    return np.roll(np.asarray(corners), -k, axis=0)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def encode(line: str) -> List[int]:
    """Characters -> class ids (0 is CTC's blank); unknown characters dropped."""
    return [CHARSET.index(c) + 1 for c in line if c in CHARSET]


def text_lines(text: str):
    """'031.0\\nPDP' -> ('031.0', 'PDP'); a sticker without an author mark has an empty second line."""
    parts = text.split("\n")
    return parts[0], parts[1] if len(parts) > 1 else ""


def greedy(log_probs: torch.Tensor) -> str:
    """Best-path decoding of (T, C) log-probabilities."""
    best = log_probs.argmax(dim=-1).tolist()
    out, prev = [], 0
    for c in best:
        if c != prev and c != 0:
            out.append(CHARSET[c - 1])
        prev = c
    return "".join(out)


def sequence_logp(log_probs: torch.Tensor, lines: Sequence[str]) -> torch.Tensor:
    """log P(line | strip) under CTC for each of N candidate lines, from one
    strip's (T, C) log-probabilities: (N,)."""
    T = log_probs.shape[0]
    targets = [torch.tensor(encode(l), dtype=torch.long) for l in lines]
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat = torch.cat(targets) if lengths.sum() > 0 else torch.zeros(0, dtype=torch.long)
    lp = log_probs[:, None].expand(T, len(lines), log_probs.shape[1])
    return -F.ctc_loss(lp, flat.to(lp.device), torch.full((len(lines),), T, dtype=torch.long), lengths,
                       reduction="none", zero_infinity=True)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def build_finder(pretrained: bool = True):
    """LR-ASPP MobileNetV3-Large with 2 classes (background, sticker core);
    from the COCO weights but for the two classifier layers."""
    from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large
    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
    model = lraspp_mobilenet_v3_large(weights=weights)
    model.classifier.low_classifier = nn.Conv2d(model.classifier.low_classifier.in_channels, 2, 1)
    model.classifier.high_classifier = nn.Conv2d(model.classifier.high_classifier.in_channels, 2, 1)
    return model


def normalised(rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float, ImageNet-normalised (as the backbone was trained)."""
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET[0]) / IMAGENET[1]
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class CRNN(nn.Module):
    """Reads one line: (N, 1, 32, 128) -> (32, N, classes) log-probabilities
    (time first, as CTC wants)."""

    def __init__(self, classes: int = len(CHARSET) + 1):
        super().__init__()

        def block(i, o, pool):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)] + \
                   ([nn.MaxPool2d(pool)] if pool else [])
        self.conv = nn.Sequential(*block(1, 32, 2), *block(32, 64, 2), *block(64, 128, None),
                                  *block(128, 128, (2, 1)), *block(128, 192, (4, 1)))
        self.rnn = nn.LSTM(192, 128, num_layers=2, bidirectional=True, batch_first=False)
        self.out = nn.Linear(256, classes)

    def forward(self, x):
        f = self.conv(x).squeeze(2).permute(2, 0, 1)          # (T, N, 192)
        return F.log_softmax(self.out(self.rnn(f)[0]), dim=-1)


def strips_tensor(strips: np.ndarray) -> torch.Tensor:
    """(N, 32, 128) uint8 -> (N, 1, 32, 128) float in [-1, 1]."""
    return torch.from_numpy(strips.astype(np.float32) / 127.5 - 1.0)[:, None]


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------
@dataclass
class Reading:
    """One sticker found in an image."""
    corners: np.ndarray             # (4, 2) px, in reading order (top-left, top-right, bottom-right, bottom-left)
    text: Optional[str]             # catalogue entry ('031.0\nPDP'), or what was read without a catalogue; None: unreadable
    confidence: float               # catalogue: the entry's share of the probability; else the reading's probability
    raw: str                        # what the network reads, best orientation, lines joined by '\n'
    char_logp: float = 0.0          # catalogue: log-probability of the entry per character

    def sure(self, min_confidence: float = MIN_CONFIDENCE, min_char_logp: float = MIN_CHAR_LOGP) -> bool:
        return self.confidence >= min_confidence and self.char_logp >= min_char_logp

    @property
    def centre(self) -> np.ndarray:
        return self.corners.mean(axis=0)


class LabelReader:
    """Finds and reads call-number stickers in a colour image (see the module
    docstring). catalogue: the texts the stickers can carry ('031.0\\nPDP'),
    as the robot knows its library's; without it, readings are the network's
    own, with their probability."""

    def __init__(self, finder: str, reader: str, catalogue: Optional[Sequence[str]] = None,
                 min_confidence: float = MIN_CONFIDENCE, min_char_logp: float = MIN_CHAR_LOGP,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.finder = build_finder(pretrained=False)
        self.finder.load_state_dict(torch.load(os.path.expanduser(finder), map_location="cpu", weights_only=False)["model"])
        self.finder.to(self.device).eval()
        self.reader = CRNN()
        self.reader.load_state_dict(torch.load(os.path.expanduser(reader), map_location="cpu", weights_only=False)["model"])
        self.reader.to(self.device).eval()
        self.catalogue = list(catalogue) if catalogue is not None else None
        self.min_confidence, self.min_char_logp = min_confidence, min_char_logp

    @torch.no_grad()
    def find(self, rgb: np.ndarray) -> List[np.ndarray]:
        """Sticker corners (4, 2), clockwise from an arbitrary corner."""
        logits = self.finder(normalised(rgb)[None].to(self.device))["out"][0]
        return quads_from_core((logits[1] > logits[0]).cpu().numpy())

    @torch.no_grad()
    def line_logp(self, gray: np.ndarray, corners: np.ndarray) -> torch.Tensor:
        """(4 orientations, 2 lines, T, C) log-probabilities."""
        strips = np.concatenate([line_strips(gray, turned(corners, k)) for k in range(4)])
        lp = self.reader(strips_tensor(strips).to(self.device))      # (T, 8, C)
        return lp.permute(1, 0, 2).reshape(4, 2, lp.shape[0], lp.shape[2]).cpu()

    def read(self, rgb: np.ndarray, corners: Optional[List[np.ndarray]] = None) -> List[Reading]:
        """Every sticker in the image (or at the given corners), read."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        quads = self.find(rgb) if corners is None else corners
        return [self.read_one(gray, q) for q in quads]

    def read_one(self, gray: np.ndarray, corners: np.ndarray) -> Reading:
        lp = self.line_logp(gray, np.asarray(corners, np.float64))
        raws = ["\n".join(greedy(lp[k, i]) for i in range(2)) for k in range(4)]
        if self.catalogue is None:
            scores = torch.stack([sequence_logp(lp[k, 0], [text_lines(raws[k])[0]])[0]
                                  + sequence_logp(lp[k, 1], [text_lines(raws[k])[1]])[0] for k in range(4)])
            k = int(scores.argmax())
            return Reading(turned(corners, k), raws[k], float(scores[k].exp()), raws[k])
        first = [text_lines(t)[0] for t in self.catalogue]
        second = [text_lines(t)[1] for t in self.catalogue]
        scores = torch.stack([sequence_logp(lp[k, 0], first) + sequence_logp(lp[k, 1], second)
                              for k in range(4)])                    # (4, entries)
        best = int(scores.argmax())
        k, e = divmod(best, len(self.catalogue))
        reading = Reading(turned(corners, k), self.catalogue[e], float(torch.softmax(scores.flatten(), 0)[best]),
                          raws[k], float(scores[k, e]) / max(1, len(first[e]) + len(second[e])))
        if not reading.sure(self.min_confidence, self.min_char_logp):
            reading.text = None
        return reading


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def read_image(root: str, stem: str):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(root, "rgb", stem + ".png")), cv2.COLOR_BGR2RGB)
    return rgb, json.load(open(os.path.join(root, "labels", stem + ".json")))


def finder_target(label: dict, shape) -> np.ndarray:
    """(H, W) uint8 target for the finder: 1 on the core of every sticker
    that is well in view, 255 (ignored) over stickers that are partly hidden
    or cut off (their outline isn't what's seen), 0 elsewhere."""
    target = np.zeros(shape, np.uint8)
    for s in sorted(label["stickers"], key=lambda s: s["seen"] >= MIN_SEEN):   # ignored ones first: cores win
        quad = np.asarray(s["corners"], np.float64)
        if s["seen"] >= MIN_SEEN:
            cv2.fillConvexPoly(target, np.round(scaled(quad, CORE) * 16).astype(np.int32), 1, cv2.LINE_8, 4)
        else:
            cv2.fillConvexPoly(target, np.round(quad * 16).astype(np.int32), 255, cv2.LINE_8, 4)
    return target


class FinderImages(torch.utils.data.Dataset):
    """make_vision_dataset.py's images with finder_target masks; `augment`
    flips left-right and varies brightness and colour a little."""

    def __init__(self, root: str, augment: bool = False, seed: int = 0):
        self.root = os.path.expanduser(root)
        self.stems = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(self.root, "labels", "*.json")))
        self.augment, self.rng = augment, np.random.default_rng(seed)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        rgb, label = read_image(self.root, self.stems[i])
        target = finder_target(label, rgb.shape[:2])
        if self.augment:
            if self.rng.random() < 0.5:
                rgb, target = rgb[:, ::-1], target[:, ::-1]
            gain = self.rng.uniform(0.75, 1.25) * self.rng.uniform(0.9, 1.1, size=3)
            rgb = np.clip(rgb * gain, 0, 255).astype(np.uint8)
        return normalised(rgb), torch.from_numpy(np.ascontiguousarray(target)).long()


def aligned(found: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The found corners started at the one nearest the true top-left, so they
    are in the true reading order."""
    return min((turned(found, k) for k in range(4)), key=lambda q: np.abs(q - truth).sum())


def reader_samples(root: str, jitters: int, seed: int = 0, min_pixels: int = 30, finder=None):
    """(strips (N, 32, 128) uint8, lines [str], views [str]) for every line
    of every sticker in view (seen >= 0.95) in the dataset at `root`: once
    from its true corners and jitters-1 times from corners moved by up to
    JITTER of its size (as the finder's corners are off); and, given a
    LabelReader as `finder`, once more from the corners its finder gives
    (what the reader sees in use)."""
    rng = np.random.default_rng(seed)
    strips, lines, views = [], [], []
    root = os.path.expanduser(root)
    stems = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(root, "labels", "*.json")))
    for n, stem in enumerate(stems):
        label = json.load(open(os.path.join(root, "labels", stem + ".json")))
        wanted = [s for s in label["stickers"] if s["seen"] >= 0.95 and s["pixels"] >= min_pixels]
        if not wanted:
            continue
        bgr = cv2.imread(os.path.join(root, "rgb", stem + ".png"))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found = finder.find(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if finder is not None else []
        for s in wanted:
            quad = np.asarray(s["corners"], np.float64)
            size = np.ptp(quad, axis=0).max()
            corners = [quad if j == 0 else quad + rng.uniform(-JITTER, JITTER, quad.shape) * size
                       for j in range(jitters)]
            corners += [aligned(f, quad) for f in found if inside(f.mean(axis=0), quad) and inside(quad.mean(axis=0), f)]
            for q in corners:
                strips.append(line_strips(gray, q))
                lines += list(text_lines(s["text"]))
                views += [label["view"]] * 2
        if n % 500 == 0:
            print(f"  {n}/{len(stems)} images, {len(lines)} lines", flush=True)
    return np.concatenate(strips) if strips else np.zeros((0,) + STRIP, np.uint8), lines, views


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_finder(data: str, out: str, epochs: int = 6, batch: int = 4, lr: float = 5e-4, workers: int = 2,
                 max_steps: Optional[int] = None):
    """Fine-tune the finder; AdamW, cosine schedule, mixed precision. Saves
    out/finder.pt after every epoch."""
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(FinderImages(data, augment=True), batch_size=batch, shuffle=True,
                                         num_workers=workers, persistent_workers=workers > 0, drop_last=True)
    model = build_finder().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * len(loader) if max_steps is None else max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    weight = torch.tensor([1.0, FINDER_WEIGHT], device=device)
    step = 0
    for epoch in range(epochs):
        model.train()
        t0, total_loss, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x)["out"], y, weight=weight, ignore_index=255)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step, n, total_loss = step + 1, n + 1, total_loss + loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total}  loss {loss.item():.4f}  {(time.time() - t0) / n:.2f} s/step",
                      flush=True)
            if step >= total:
                break
        torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(out, "finder.pt"))
        print({"epoch": epoch, "loss": total_loss / max(n, 1), "seconds": round(time.time() - t0)}, flush=True)
        if step >= total:
            break


def train_reader(data: str, out: str, val: Optional[str] = None, epochs: int = 20, batch: int = 256,
                 lr: float = 1e-3, jitters: int = 2):
    """Train the CRNN on reader_samples (cached in out/reader_cache.npz),
    with the finder's own crops too if out/finder.pt exists (train the finder
    first); Adam, cosine schedule; light augmentation (contrast, brightness,
    blur). Saves out/reader.pt; prints line accuracy on `val` per epoch."""
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cache = os.path.join(out, "reader_cache.npz")
    if not os.path.exists(cache):
        print("cutting out sticker lines ...", flush=True)
        finder = None
        if os.path.exists(os.path.join(out, "finder.pt")):
            finder = LabelReader.__new__(LabelReader)
            finder.device = device
            finder.finder = build_finder(pretrained=False)
            finder.finder.load_state_dict(torch.load(os.path.join(out, "finder.pt"), map_location="cpu",
                                                     weights_only=False)["model"])
            finder.finder.to(device).eval()
        strips, lines, _ = reader_samples(data, jitters, finder=finder)
        del finder
        np.savez(cache, strips=strips, lines=np.array(lines))
    z = np.load(cache)
    strips, lines = z["strips"], list(z["lines"])
    targets = [torch.tensor(encode(l), dtype=torch.long) for l in lines]
    val_set = reader_samples(val, 1) if val else None
    print(f"{len(lines)} training lines", flush=True)
    model = CRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * math.ceil(len(lines) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(lines))
        t0, total = time.time(), 0.0
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            x = strips[idx].astype(np.float32)
            contrast = rng.uniform(0.6, 1.2, (len(idx), 1, 1))
            x = np.clip((x - x.mean(axis=(1, 2), keepdims=True)) * contrast + x.mean(axis=(1, 2), keepdims=True)
                        + rng.uniform(-30, 30, (len(idx), 1, 1)), 0, 255)
            blur = rng.random(len(idx)) < 0.3
            if blur.any():
                x[blur] = np.stack([cv2.GaussianBlur(s, (3, 3), 0) for s in x[blur]])
            xt = torch.from_numpy(x / 127.5 - 1.0).float()[:, None].to(device)
            tg = [targets[i] for i in idx]
            lp = model(xt)
            loss = F.ctc_loss(lp, torch.cat(tg).to(device), torch.full((len(idx),), lp.shape[0], dtype=torch.long),
                              torch.tensor([len(t) for t in tg], dtype=torch.long), zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(out, "reader.pt"))
        msg = f"epoch {epoch}: loss {total / len(lines):.4f}, {time.time() - t0:.0f} s"
        if val_set is not None:
            msg += f", val lines right {100 * line_accuracy(model, val_set[0], val_set[1], device):.1f}%"
        print(msg, flush=True)


@torch.no_grad()
def line_accuracy(model, strips, lines, device) -> float:
    model.eval()
    right = 0
    for b in range(0, len(lines), 512):
        lp = model(strips_tensor(strips[b:b + 512]).to(device)).cpu()
        right += sum(greedy(lp[:, i]) == lines[b + i] for i in range(lp.shape[1]))
    model.train()
    return right / max(len(lines), 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def sticker_distance(depth_png: np.ndarray, corners, unit: float) -> float:
    """Median depth (m) over a sticker's pixels."""
    m = np.zeros(depth_png.shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(np.asarray(corners)).astype(np.int32), 1)
    d = depth_png[(m > 0) & (depth_png > 0)]
    return float(np.median(d)) * unit if len(d) else float("nan")


def inside(point, quad) -> bool:
    return cv2.pointPolygonTest(np.asarray(quad, np.float32).reshape(-1, 1, 2), (float(point[0]), float(point[1])), False) >= 0


def evaluate(root: str, weights: str, min_confidence: float = MIN_CONFIDENCE):
    """Over the images in `root` with stickers: per camera (kind of view) and
    distance, of the stickers in view (seen >= 0.95), the share found, read
    right (matched to the right catalogue entry), read wrong, and unreadable;
    the catalogue is every sticker text seen in that scene. Then the share
    read right and wrong at other confidence thresholds."""
    from learned_segmenter import view_group
    root = os.path.expanduser(root)
    reader = LabelReader(os.path.join(weights, "finder.pt"), os.path.join(weights, "reader.pt"),
                         min_confidence=0.0, min_char_logp=-1e9)
    stems = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(root, "labels", "*.json")))
    by_scene = collections.defaultdict(set)
    labels = {}
    for stem in stems:
        labels[stem] = json.load(open(os.path.join(root, "labels", stem + ".json")))
        for s in labels[stem]["stickers"]:
            by_scene[labels[stem]["seed"]].add(s["text"])
    rows = []                        # (view, distance, found, truth, reading or None)
    false_finds, seconds, images = 0, 0.0, 0
    for stem in stems:
        label = labels[stem]
        if not label["stickers"]:
            continue
        rgb, _ = read_image(root, stem)
        depth = cv2.imread(os.path.join(root, "depth", stem + ".png"), cv2.IMREAD_UNCHANGED)
        reader.catalogue = sorted(by_scene[label["seed"]])
        t0 = time.time()
        readings = reader.read(rgb)
        seconds, images = seconds + time.time() - t0, images + 1
        used = set()
        for s in label["stickers"]:
            if s["seen"] < 0.95:
                continue
            centre = np.mean(s["corners"], axis=0)
            hit = next((i for i, r in enumerate(readings) if i not in used and inside(centre, r.corners)
                        and inside(r.centre, s["corners"])), None)
            if hit is not None:
                used.add(hit)
            rows.append((view_group(label["view"]), sticker_distance(depth, s["corners"], label["depth_unit"]),
                         s["text"], None if hit is None else readings[hit]))
        whole = [s["corners"] for s in label["stickers"]]
        false_finds += sum(1 for i, r in enumerate(readings) if i not in used
                           and not any(inside(r.centre, q) for q in whole))
    print(f"{images} images, {seconds / max(images, 1):.2f} s an image (find + read); "
          f"{false_finds} found stickers that are no sticker\n")
    print(f"confidence >= {min_confidence}:")
    print(f"{'camera':12s}{'distance m':>12s}{'stickers':>10s}{'found':>8s}{'right':>8s}{'wrong':>8s}{'unreadable':>12s}")
    groups = collections.defaultdict(list)
    for view, dist, truth, r in rows:
        b = next((i for i in range(len(DISTANCE_BINS) - 1) if DISTANCE_BINS[i] <= dist < DISTANCE_BINS[i + 1]), -1)
        groups[(view, b)].append((truth, r))          # -1: no depth reading on the sticker (white, far, or glare)
        groups[("all", None)].append((truth, r))
    for (view, b), items in sorted(groups.items(), key=lambda kv: (kv[0][0] == "all", kv[0][0], kv[0][1] or 0)):
        n = len(items)
        found = sum(r is not None for _, r in items)
        right = sum(r is not None and r.sure(min_confidence) and r.text == t for t, r in items)
        wrong = sum(r is not None and r.sure(min_confidence) and r.text != t for t, r in items)
        dist = "" if b is None else "no depth" if b < 0 else f"{DISTANCE_BINS[b]:.2f}-{DISTANCE_BINS[b + 1]:.2f}"
        print(f"{view:12s}{dist:>12s}{n:10d}{100 * found / n:7.1f}%{100 * right / n:7.1f}%{100 * wrong / n:7.1f}%"
              f"{100 * (found - right - wrong) / n:11.1f}%")
    found = [(t, r) for _, _, t, r in rows if r is not None]
    print("\nfound stickers, by confidence threshold: read right / read wrong / unreadable")
    for c in (0.0, 0.5, 0.9, 0.99, 0.999):
        right = sum(r.sure(c) and r.text == t for t, r in found)
        wrong = sum(r.sure(c) and r.text != t for t, r in found)
        print(f"  >= {c:<6}  {100 * right / len(found):5.1f}% / {100 * wrong / len(found):5.2f}% / "
              f"{100 * (len(found) - right - wrong) / len(found):5.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("train-finder", "train-reader", "eval"))
    ap.add_argument("--data", required=True, help="dataset folder (make_vision_dataset.py --out)")
    ap.add_argument("--val", help="train-reader: validation folder")
    ap.add_argument("--out", default="labels", help="train: folder for finder.pt / reader.pt")
    ap.add_argument("--weights", help="eval: folder with finder.pt and reader.pt")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-steps", type=int, help="train-finder: stop after this many steps (a quick check)")
    ap.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    if a.command == "train-finder":
        train_finder(a.data, out, epochs=a.epochs or 6, max_steps=a.max_steps)
    elif a.command == "train-reader":
        train_reader(a.data, out, a.val, epochs=a.epochs or 20)
    else:
        evaluate(a.data, os.path.expanduser(a.weights), a.min_confidence)


if __name__ == "__main__":
    main()
