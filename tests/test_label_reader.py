"""Checks of label_reader's geometry and text plumbing and of cart_demo's
learned-label helpers, without trained networks (no training here).

    uv run pytest tests/test_label_reader.py      # needs `uv sync --group rl`
"""

import math

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from bila.control import cart_demo  # noqa: E402
from bila.vision import label_reader as lr  # noqa: E402


def quad(cx, cy, w, h, angle_deg):
    """Corners of a w x h rectangle about (cx, cy), turned, in reading order."""
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return local @ R.T + (cx, cy)


def same_corners(a, b, tol):
    """a equals b up to where the corner list starts."""
    return min(np.abs(np.roll(a, k, axis=0) - b).max() for k in range(4)) < tol


def test_finder_target_marks_cores_and_ignores_hidden_stickers():
    seen, hidden = quad(60, 80, 30, 80, 10), quad(160, 80, 30, 80, 0)
    label = {"stickers": [{"corners": seen.tolist(), "seen": 1.0}, {"corners": hidden.tolist(), "seen": 0.5}]}
    t = lr.finder_target(label, (160, 220))
    assert (t == 1).sum() == pytest.approx(lr.CORE ** 2 * 30 * 80, rel=0.08)
    assert (t == 255).sum() == pytest.approx(30 * 80, rel=0.08)
    assert t[80, 60] == 1 and t[80, 160] == 255 and t[5, 5] == 0


def test_quads_from_core_gives_back_the_stickers():
    stickers = [quad(60, 80, 30, 80, 10), quad(120, 80, 24, 70, -5), quad(190, 90, 60, 30, 30)]
    label = {"stickers": [{"corners": q.tolist(), "seen": 1.0} for q in stickers]}
    found = lr.quads_from_core(lr.finder_target(label, (180, 260)) == 1)
    assert len(found) == 3
    for q in found:
        assert any(same_corners(q, s, 3.0) for s in stickers)
        assert np.array_equal(q, lr.clockwise(q))


def test_clockwise_and_turned():
    q = quad(0, 0, 2, 4, 0)                                     # reading order: clockwise on screen
    assert np.array_equal(lr.clockwise(q), q)
    assert np.array_equal(lr.clockwise(q[::-1]), q)
    assert np.array_equal(lr.turned(q, 1)[0], q[1]) and np.array_equal(lr.turned(q, 4), q)


def test_line_strips_cut_the_lines_wherever_the_sticker_is():
    """A sticker texture warped into an image reads the same as the flat texture."""
    tex = cv2.cvtColor(cv2.imread(cart_demo.sticker(315, "PDP", 0.4)), cv2.COLOR_BGR2GRAY)
    h, w = tex.shape
    flat = lr.line_strips(tex, np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    corners = np.array([[110, 40], [190, 55], [185, 260], [100, 250]], np.float32)
    P = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), corners)
    image = cv2.warpPerspective(tex, P, (320, 320), borderValue=90)
    strips = lr.line_strips(image, corners)
    assert strips.shape == (2,) + lr.STRIP
    for a, b in zip(flat, strips):
        assert np.corrcoef(a.ravel().astype(float), b.ravel().astype(float))[0, 1] > 0.9
    # Read from another corner first, it is another image (the sticker turned).
    turned = lr.line_strips(image, lr.turned(corners, 2))
    assert np.corrcoef(flat[0].ravel().astype(float), turned[0].ravel().astype(float))[0, 1] < 0.7   # (white page alike)


def peaked(text, T=32, sure=8.0):
    """(T, C) log-probabilities that spell `text` (blanks in between)."""
    ids = []
    for c in lr.encode(text):
        ids += [c, 0]
    ids = (ids + [0] * T)[:T]
    logits = np.zeros((T, len(lr.CHARSET) + 1), np.float32)
    logits[np.arange(T), ids] = sure
    return torch.log_softmax(torch.tensor(logits), dim=-1)


def test_text_decoding_and_catalogue_scores():
    lp = peaked("031.0")
    assert lr.greedy(lp) == "031.0"
    scores = lr.sequence_logp(lp, ["031.0", "031.5", "013.0", ""])
    assert int(scores.argmax()) == 0 and scores[0] > scores[1] + 5
    assert lr.text_lines("031.0\nPDP") == ("031.0", "PDP") and lr.text_lines("031.0") == ("031.0", "")
    assert lr.encode("a1?") == [lr.CHARSET.index("1") + 1]


def test_reading_is_sure_only_above_both_thresholds():
    r = lr.Reading(np.zeros((4, 2)), "031.0\nPDP", 0.995, "031.0\nPDP", -0.2)
    assert r.sure() and not r.sure(min_confidence=0.999) and not r.sure(min_char_logp=-0.1)


def test_crnn_shapes():
    out = lr.CRNN()(torch.zeros(3, 1, *lr.STRIP))
    assert out.shape == (32, 3, len(lr.CHARSET) + 1)


# --- cart_demo's side -------------------------------------------------------
def test_sticker_text_matches_the_dataset():
    assert cart_demo.sticker_text(315, "PDP") == "031.5\nPDP"
    assert cart_demo.sticker_text(310, "") == "031.0"


def test_bracket():
    class B:
        def __init__(self, name):
            self.name = name

    class R:
        books = [B("a"), B("b"), B("c"), B("d")]

    read = {"a": 100, "b": 110, "d": 130}
    assert cart_demo.bracket(R, read, 115) == (1, 3)           # c unread: not side by side
    assert cart_demo.bracket(R, read, 105) == (0, 1)
    assert cart_demo.bracket(R, read, 95) == (None, 0)


def test_locate_finds_the_book_and_the_sticker_in_the_world():
    class Frame:
        pass
    f = Frame()
    f.depth = np.full((40, 60), 0.5, np.float32)
    f.seg = np.full((40, 60), -1, np.int32)
    f.seg[:, 30:] = 7
    f.K = np.array([[50.0, 0, 30], [0, 50.0, 20], [0, 0, 1]])
    f.pose = np.eye(4)
    f.pose[:3, 3] = (1.0, 2.0, 3.0)
    labels = cart_demo.LearnedLabels.__new__(cart_demo.LearnedLabels)
    labels.book_of = {7: "book_3"}
    book, point = labels.locate(f, np.array([[34, 10], [44, 10], [44, 30], [34, 30]], float))
    assert book == "book_3"
    assert point == pytest.approx([1.0 + (39 - 30) * 0.5 / 50, 2.0 + 0.0, 3.5])
    assert labels.locate(f, np.array([[2, 2], [8, 2], [8, 8], [2, 8]], float)) == (None, None)
