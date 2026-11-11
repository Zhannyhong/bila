"""
A learned book segmenter: Mask R-CNN (torchvision, fine-tuned from COCO) that
finds one mask per book in a colour image, in place of OracleSegmenter's
reading of the simulation's object ids.

    from bila.vision.learned_segmenter import LearnedSegmenter
    with RGBDPerception(model, row, segmenter=LearnedSegmenter("segmenter.pt")) as perception:
        est = perception.estimate(data)

    # in WSL, on the dataset make_vision_dataset.py writes (needs `uv sync --group rl`)
    MUJOCO_GL=egl uv run python -m bila.vision.learned_segmenter train --data ~/bila_data/train --val ~/bila_data/val \
        --out ~/bila_data/segmenter
    uv run python -m bila.vision.learned_segmenter eval --data ~/bila_data/val --weights ~/bila_data/segmenter/segmenter.pt

It sees only the colour image (depth comes in afterwards, in RGBDPerception's
spine fits), so the COCO-pretrained backbone is used as it is. There is one
class, "book": every book in the image, on any level, in the tray or held.

`eval` scores it against the oracle's masks on the validation images: each
true book is matched to the predicted mask it overlaps most (one to one, best
first), and the tables give the share found (IoU >= 0.5), the share of
predictions that are a book, and the IoU of the matches, by kind of view and
by where the book is.
"""

from __future__ import annotations

# Rendering (MuJoCo's EGL) after torch is imported crashes unless this is
# imported first; whatever imports this module may render.
import torch._dynamo  # noqa: F401  (see above)

import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from bila.vision.rgbd_perception import OracleSegmenter, Segment

MIN_TRAIN_PIXELS = 40           # a book showing fewer pixels than this isn't a training target (a sliver at an edge)
SCORE = 0.5                     # detections less sure than this are dropped
MASK_THRESHOLD = 0.5            # a pixel is in a mask where its probability is above this
MIN_PIXELS = 30                 # a mask left with fewer pixels than this (once overlaps are resolved) is dropped
MAX_BOOKS = 150                 # detections kept per image (images have up to ~115 books)
ASPECTS = (0.5, 1.0, 2.0, 4.0)  # anchor height / width: spines seen front on are tall and thin
MATCH_IOU = 0.5                 # a prediction this close to a true book counts as finding it


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def default_device() -> str:
    """An NVIDIA GPU, else Apple's (MPS: full precision only), else the CPU."""
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def build_model(pretrained: bool = True):
    """Mask R-CNN (ResNet-50 FPN) for one class, at the images' own size
    (1280 x 720: the default would scale them up to 800 high). Starts from the
    COCO weights except for the layers whose shape changes: the box and mask
    classifiers (2 classes, not 91) and the region proposals' outputs (an
    extra, taller anchor shape)."""
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.rpn import RPNHead

    anchors = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)), aspect_ratios=(ASPECTS,) * 5)
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2, min_size=720, max_size=1280,
                                  rpn_anchor_generator=anchors, rpn_head=RPNHead(256, len(ASPECTS)),
                                  box_detections_per_img=MAX_BOOKS, box_score_thresh=0.05)
    if pretrained:
        coco = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=False)
        own = model.state_dict()
        model.load_state_dict({k: v for k, v in coco.items() if k in own and own[k].shape == v.shape}, strict=False)
    return model


def load_model(weights: str, device: str):
    model = build_model(pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0


def instance_map(output: Dict[str, torch.Tensor], shape, score: float = SCORE,
                 mask_threshold: float = MASK_THRESHOLD, min_pixels: int = MIN_PIXELS) -> np.ndarray:
    """(H, W) int32: 0 for no book, k for the k-th book kept, from one image's
    Mask R-CNN output. Masks can overlap; a pixel goes to the surest detection
    that claims it (so every pixel is in at most one book's mask, as with the
    oracle). Masks left with fewer than min_pixels pixels are dropped."""
    keep = output["scores"] >= score
    scores = output["scores"][keep].cpu().numpy()
    probs = output["masks"][keep, 0]
    out = np.zeros(shape, dtype=np.int32)
    taken = torch.zeros(shape, dtype=torch.bool, device=probs.device)
    k = 0
    for i in np.argsort(-scores, kind="stable"):
        mask = (probs[i] > mask_threshold) & ~taken
        if int(mask.sum()) < min_pixels:
            continue
        taken |= mask
        k += 1
        out[mask.cpu().numpy()] = k
    return out


class LearnedSegmenter:
    """Book masks from the colour image by a trained Mask R-CNN; the same
    interface as OracleSegmenter: segmenter(frame) -> [Segment]. Segments
    carry no label (it doesn't know which simulated book is which). Use it
    with RGBDPerception(robust_fit=True): its masks bleed a few pixels onto
    the next book, which the default spine fit takes for the book's edge.
    device None: the GPU if there is one (default_device)."""

    def __init__(self, weights: str, device: Optional[str] = None, score: float = SCORE,
                 mask_threshold: float = MASK_THRESHOLD):
        self.device = device or default_device()
        self.model = load_model(weights, self.device)
        self.score, self.mask_threshold = score, mask_threshold

    @torch.no_grad()
    def instances(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W) int32 book index per pixel (0 = none), as instance_map."""
        output = self.model([to_tensor(rgb).to(self.device)])[0]
        return instance_map(output, rgb.shape[:2], self.score, self.mask_threshold)

    def __call__(self, frame) -> List[Segment]:
        books = self.instances(frame.rgb)
        return [Segment(books == k) for k in range(1, int(books.max()) + 1)]


class OracleLabelled:
    """A segmenter whose segments are given the label of the oracle's book
    they overlap most (IoU >= MATCH_IOU, one to one), else None. Only for
    checking: it lets evaluations match learned books to true ones."""

    def __init__(self, segmenter, model):
        self.segmenter, self.oracle = segmenter, OracleSegmenter(model)

    def __call__(self, frame) -> List[Segment]:
        segs = self.segmenter(frame)
        truth = self.oracle(frame)
        pred = np.zeros(frame.depth.shape, dtype=np.int32)
        true = np.zeros(frame.depth.shape, dtype=np.int32)
        for k, s in enumerate(segs, 1):
            pred[s.mask] = k
        for k, s in enumerate(truth, 1):
            true[s.mask] = k
        for p, t, _ in match(pred, true):
            segs[p - 1].label = truth[t - 1].label
        return segs


# ---------------------------------------------------------------------------
# Matching (for the evaluation)
# ---------------------------------------------------------------------------
def iou_table(pred: np.ndarray, true: np.ndarray):
    """IoU of every predicted instance (rows, 1..P) with every true one
    (columns, 1..T), from two instance maps (0 = none)."""
    P, T = int(pred.max()), int(true.max())
    both = np.bincount(pred.ravel() * (T + 1) + true.ravel(), minlength=(P + 1) * (T + 1)).reshape(P + 1, T + 1)
    area_p, area_t = both.sum(axis=1), both.sum(axis=0)
    inter = both[1:, 1:].astype(np.float64)
    union = area_p[1:, None] + area_t[None, 1:] - inter
    return inter / np.maximum(union, 1)


def match(pred: np.ndarray, true: np.ndarray, min_iou: float = MATCH_IOU):
    """[(predicted k, true k, IoU)] one to one, best pairs first, IoU >= min_iou."""
    iou = iou_table(pred, true)
    out = []
    if iou.size == 0:
        return out
    order = np.argsort(-iou, axis=None)
    used_p, used_t = set(), set()
    for flat in order:
        p, t = np.unravel_index(flat, iou.shape)
        if iou[p, t] < min_iou:
            break
        if p in used_p or t in used_t:
            continue
        used_p.add(p), used_t.add(t)
        out.append((int(p) + 1, int(t) + 1, float(iou[p, t])))
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def view_group(view: str) -> str:
    """'sweep+0.3' -> 'sweep', 'close_row1' -> 'close_row', ..."""
    v = view.rstrip("0123456789").rstrip("+-.")
    return "sweep" if v.startswith("sweep") else v


class BookImages(torch.utils.data.Dataset):
    """make_vision_dataset.py's images as Mask R-CNN training samples: the
    colour image and, per book showing at least MIN_TRAIN_PIXELS, its mask and
    box. `augment` flips half the images left-right and varies brightness,
    contrast and colour a little (the lighting varies in the scenes too)."""

    def __init__(self, root: str, augment: bool = False, seed: int = 0):
        self.root = os.path.expanduser(root)
        self.stems = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(self.root, "labels", "*.json")))
        self.augment, self.rng = augment, np.random.default_rng(seed)

    def __len__(self):
        return len(self.stems)

    def read(self, i: int):
        stem = self.stems[i]
        rgb = cv2.cvtColor(cv2.imread(os.path.join(self.root, "rgb", stem + ".png")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.root, "masks", stem + ".png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        return stem, rgb, mask

    def __getitem__(self, i: int):
        _, rgb, mask = self.read(i)
        if self.augment:
            if self.rng.random() < 0.5:
                rgb, mask = rgb[:, ::-1], mask[:, ::-1]
            gain = self.rng.uniform(0.75, 1.25) * self.rng.uniform(0.9, 1.1, size=3)
            mean = rgb.reshape(-1, 3).mean(axis=0)
            contrast = self.rng.uniform(0.8, 1.2)
            rgb = np.clip((rgb - mean) * contrast + mean * gain, 0, 255).astype(np.uint8)
        ids, counts = np.unique(mask, return_counts=True)
        ids = ids[(ids > 0) & (counts >= MIN_TRAIN_PIXELS)]
        masks = np.stack([mask == k for k in ids]) if len(ids) else np.zeros((0,) + mask.shape, dtype=bool)
        boxes = np.zeros((len(ids), 4), dtype=np.float32)
        for j, m in enumerate(masks):
            v, u = np.nonzero(m)
            boxes[j] = (u.min(), v.min(), u.max() + 1, v.max() + 1)
        target = {"boxes": torch.from_numpy(boxes), "labels": torch.ones(len(ids), dtype=torch.int64),
                  "masks": torch.from_numpy(masks.astype(np.uint8))}
        return to_tensor(rgb), target


def collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(data: str, out: str, val: Optional[str] = None, epochs: int = 8, batch: int = 2, lr: float = 0.01,
          workers: int = 2, max_steps: Optional[int] = None):
    """Fine-tune on `data`; SGD with a short warm-up, then the learning rate
    falls along a cosine to zero; mixed precision (the GPU has 4 GB). Saves
    out/segmenter.pt after every epoch (with the optimiser, to resume from)
    and logs the losses and, if `val` is given, the validation scores."""
    os.makedirs(out, exist_ok=True)
    device = default_device()
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(BookImages(data, augment=True), batch_size=batch, shuffle=True,
                                         num_workers=workers, collate_fn=collate, persistent_workers=workers > 0)
    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total = epochs * len(loader) if max_steps is None else max_steps
    warmup = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1))))
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    path, start, step = os.path.join(out, "segmenter.pt"), 0, 0
    if os.path.exists(path):                                       # resume
        state = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]), opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"]), scaler.load_state_dict(state["scaler"])
        start, step = state["epoch"] + 1, state["step"]
        print(f"resuming after epoch {state['epoch']}", flush=True)
    log = open(os.path.join(out, "train_log.jsonl"), "a")
    for epoch in range(start, epochs):
        model.train()
        t0, sums, n = time.time(), {}, 0
        for images, targets in loader:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast(device, enabled=device == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step, n = step + 1, n + 1
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            if step % 50 == 0:
                gpu = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
                print(f"epoch {epoch} step {step}/{total}  loss {loss.item():.3f}  lr {sched.get_last_lr()[0]:.4f}  "
                      f"{(time.time() - t0) / n:.2f} s/step  gpu peak {gpu:.2f} GB", flush=True)
            if max_steps is not None and step >= max_steps:
                break
        record = {"epoch": epoch, "step": step, "seconds": time.time() - t0, **{k: v / n for k, v in sums.items()}}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "step": step}, path)
        if val:
            scores = evaluate(model, val, device, max_images=100)
            record.update({k: v for k, v in scores["all"].items()})
            model.train()
        log.write(json.dumps(record) + "\n"), log.flush()
        print(record, flush=True)
        if max_steps is not None and step >= max_steps:
            break


# ---------------------------------------------------------------------------
# Evaluation against the oracle's masks
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, root: str, device: str, max_images: Optional[int] = None, by: bool = False):
    """Scores over the images in `root` (see the module docstring): {group:
    {recall, precision, iou, books, images}} for 'all' and, with by=True,
    per kind of view, per where the book is (target_row, other_level, tray,
    held), and for the target row's books that show at least 2000 pixels."""
    model.eval()
    ds = BookImages(root)
    stems = ds.stems if max_images is None else ds.stems[:: max(1, len(ds.stems) // max_images)][:max_images]
    tally: Dict[str, Dict[str, float]] = {}

    def add(group, found, books, true_pos, preds, iou_sum, image=0):
        t = tally.setdefault(group, dict(found=0, books=0, true_pos=0, preds=0, iou=0.0, images=0))
        t["found"] += found; t["books"] += books; t["true_pos"] += true_pos
        t["preds"] += preds; t["iou"] += iou_sum; t["images"] += image

    seconds = 0.0
    for stem in stems:
        _, rgb, true = ds.read(ds.stems.index(stem))
        t0 = time.time()
        pred = instance_map(model([to_tensor(rgb).to(device)])[0], rgb.shape[:2])
        seconds += time.time() - t0
        pairs = match(pred, true)
        found = {t: iou for _, t, iou in pairs}
        n_pred = int(pred.max())
        add("all", len(pairs), int(true.max()), len(pairs), n_pred, sum(found.values()), 1)
        if not by:
            continue
        label = json.load(open(os.path.join(ds.root, "labels", stem + ".json")))
        view = view_group(label["view"])
        add(view, len(pairs), int(true.max()), len(pairs), n_pred, sum(found.values()), 1)
        for k, b in enumerate(label["books"], 1):
            hit = k in found
            groups = [b["where"]] + (["target_row, >= 2000 px"] if b["where"] == "target_row" and b["pixels"] >= 2000 else [])
            for g in groups:
                add(g, int(hit), 1, 0, 0, found.get(k, 0.0))
    out = {}
    for g, t in tally.items():
        out[g] = dict(recall=t["found"] / max(t["books"], 1), iou=t["iou"] / max(t["found"], 1), books=t["books"],
                      images=t["images"])
        if t["preds"]:
            out[g]["precision"] = t["true_pos"] / t["preds"]
    out["all"]["seconds_per_image"] = seconds / max(len(stems), 1)
    return out


def print_scores(scores):
    print(f"{'':26s}{'found (IoU>=0.5)':>18s}{'precision':>11s}{'IoU of found':>14s}{'books':>8s}{'images':>8s}")
    for g, s in scores.items():
        prec = f"{100 * s['precision']:.1f}%" if "precision" in s else "-"
        print(f"{g:26s}{100 * s['recall']:17.1f}%{prec:>11s}{s['iou']:14.3f}{s['books']:8d}{s['images'] or '':>8}")
    print(f"{scores['all']['seconds_per_image']:.2f} s an image")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("train", "eval"))
    ap.add_argument("--data", required=True, help="dataset folder (make_vision_dataset.py --out)")
    ap.add_argument("--val", help="train: validation folder, scored after every epoch")
    ap.add_argument("--out", default="segmenter", help="train: folder for segmenter.pt and the log")
    ap.add_argument("--weights", help="eval: segmenter.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=2, help="train: data-loading processes")
    ap.add_argument("--max-steps", type=int, help="train: stop after this many steps (a quick check)")
    a = ap.parse_args()
    if a.command == "train":
        train(a.data, os.path.expanduser(a.out), a.val, a.epochs, a.batch, a.lr, a.workers, a.max_steps)
    else:
        device = default_device()
        print_scores(evaluate(load_model(os.path.expanduser(a.weights), device), a.data, device, by=True))


if __name__ == "__main__":
    main()
