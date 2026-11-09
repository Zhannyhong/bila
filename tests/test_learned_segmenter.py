"""Checks of learned_segmenter's plumbing, with an untrained model (no
training here): overlap resolution, matching, the dataset reader, the
segmenter interface and the oracle labelling the evaluations use.

    MUJOCO_GL=egl uv run pytest tests/test_learned_segmenter.py      # in WSL; needs `uv sync --group rl`
"""

import json

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from bila.vision import learned_segmenter as ls  # noqa: E402
from bila.vision.rgbd_perception import Segment  # noqa: E402


def fake_output(masks, scores):
    """A Mask R-CNN output dict for given (N, H, W) probabilities and scores."""
    return {"masks": torch.tensor(np.asarray(masks, dtype=np.float32))[:, None],
            "scores": torch.tensor(scores, dtype=torch.float32)}


def test_instance_map_gives_overlaps_to_the_surer_mask_and_drops_slivers():
    a = np.zeros((20, 30)); a[:, :15] = 0.9
    b = np.zeros((20, 30)); b[:, 10:] = 0.8
    c = np.zeros((20, 30)); c[0, 0] = 1.0                  # 1 pixel
    weak = np.ones((20, 30))
    out = ls.instance_map(fake_output([b, a, c, weak], [0.7, 0.9, 0.95, 0.2]), (20, 30), min_pixels=5)
    assert out.max() == 2
    assert (out[:, :15] == 1).all() and (out[:, 15:] == 2).all()   # a (surer) keeps the overlap


def test_match_is_one_to_one_and_best_first():
    true = np.zeros((10, 10), int); true[:, :5] = 1; true[:, 5:] = 2
    pred = np.zeros((10, 10), int); pred[:, :4] = 1; pred[:, 4:] = 2; pred[0, 9] = 3
    pairs = ls.match(pred, true)
    assert sorted((p, t) for p, t, _ in pairs) == [(1, 1), (2, 2)]
    assert all(iou >= ls.MATCH_IOU for *_, iou in pairs)
    iou = ls.iou_table(pred, true)
    assert iou.shape == (3, 2) and iou[0, 0] == pytest.approx(0.8)


def test_view_group():
    assert [ls.view_group(v) for v in ("sweep+0.3", "sweep-0.3", "sweep+0.0", "random0", "close_row1", "row_cam")] \
        == ["sweep", "sweep", "sweep", "random", "close_row", "row_cam"]


def test_book_images_reads_masks_and_boxes(tmp_path):
    for sub in ("rgb", "masks", "labels"):
        (tmp_path / sub).mkdir()
    rgb = np.zeros((40, 60, 3), np.uint8); rgb[..., 2] = 200          # blue, in RGB
    mask = np.zeros((40, 60), np.uint16)
    mask[5:35, 10:20] = 1
    mask[5:35, 30:45] = 2
    mask[0, 0:3] = 3                                                    # too few pixels to train on
    cv2.imwrite(str(tmp_path / "rgb" / "a.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(tmp_path / "masks" / "a.png"), mask)
    (tmp_path / "labels" / "a.json").write_text(json.dumps({"image": "a", "view": "random0", "books": []}))
    image, target = ls.BookImages(str(tmp_path))[0]
    assert image.shape == (3, 40, 60) and float(image[2].max()) == pytest.approx(200 / 255)
    assert target["boxes"].tolist() == [[10, 5, 20, 35], [30, 5, 45, 35]]
    assert target["masks"].shape == (2, 40, 60) and int(target["masks"][1].sum()) == 30 * 15
    assert target["labels"].tolist() == [1, 1]


class Frame:
    def __init__(self, rgb):
        self.rgb, self.depth = rgb, np.ones(rgb.shape[:2], np.float32)


def test_learned_segmenter_interface(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "segmenter.pt"
    torch.save({"model": ls.build_model(pretrained=False).state_dict()}, path)
    seg = ls.LearnedSegmenter(str(path), device="cpu", score=0.0)       # untrained: keep whatever it finds
    rgb = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    segs = seg(Frame(rgb))
    assert all(isinstance(s, Segment) and s.mask.shape == (120, 200) and s.mask.dtype == bool and s.label is None
               for s in segs)
    covered = np.sum([s.mask for s in segs], axis=0) if segs else np.zeros((120, 200))
    assert covered.max() <= 1                                           # masks don't overlap


def test_oracle_labelled_takes_the_overlapping_books_label():
    class Oracle:
        def __call__(self, frame):
            a = np.zeros((10, 10), bool); a[:, :5] = True
            return [Segment(a, 7), Segment(~a, 9)]

    class Learned:
        def __call__(self, frame):
            a = np.zeros((10, 10), bool); a[:, :6] = True
            b = np.zeros((10, 10), bool); b[0, 9] = True
            return [Segment(~a & ~b), Segment(a), Segment(b)]

    labelled = ls.OracleLabelled.__new__(ls.OracleLabelled)
    labelled.segmenter, labelled.oracle = Learned(), Oracle()
    segs = labelled(Frame(np.zeros((10, 10, 3), np.uint8)))
    assert [s.label for s in segs] == [9, 7, None]
