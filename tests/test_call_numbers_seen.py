import numpy as np
import pytest

from bila.vision.call_numbers_seen import Sticker, slot_between


def s(number, x):
    return Sticker(number, x, np.zeros((4, 2)))


def test_brackets_the_number_and_merges_repeats():
    # numbers go up toward -x (left to right seen from the front); 135 belongs between 130 and 140
    seen = [s(120, 0.10), s(130, 0.06), s(140, 0.00), s(150, -0.04), s(130, 0.08)]
    lo, hi = slot_between(seen, 135)
    assert (lo.number, hi.number) == (130, 140)
    assert abs(lo.x - 0.07) < 1e-9                       # read twice: averaged


def test_none_without_both_sides():
    assert slot_between([s(120, 0.1), s(130, 0.0)], 135) is None
    assert slot_between([], 135) is None


def test_a_read_sticker_sizes_its_book_from_the_catalogue():
    from types import SimpleNamespace
    from bila.vision.call_numbers_seen import sized, sticker_on, with_catalogue
    from bila.vision.perception import ShelfEstimate, book_from_spine
    # the camera's fit: upright, 24 mm wide (too thick), 0.2 m tall (its top cut off), depth assumed
    fit = book_from_spine([[0, 0.1, 0], [0.024, 0.1, 0], [0.024, 0.1, 0.2], [0, 0.1, 0.2]], -1, 0.15, 0.8)
    on = Sticker(130, 0.010, np.zeros((4, 2)), np.array([0.010, 0.1, 0.045]))
    off = Sticker(140, 0.040, np.zeros((4, 2)), np.array([0.040, 0.1, 0.045]))
    assert sticker_on(fit, [off, on]) is on
    record = SimpleNamespace(thickness=0.020, height=0.25, depth=0.18, mass=0.6)
    b = sized(fit, on, record)
    assert np.allclose(b.spine_corners[[0, 1], 0], [0.0, 0.020])        # centred on the sticker, the record's width
    assert (b.thickness, b.height, b.depth, b.mass, b.call_number) == pytest.approx((0.020, 0.25, 0.18, 0.6, 130))
    assert b.slot_offset == -1 and b.visible == 0.8
    est = ShelfEstimate(np.eye(4), 0, 0.1, 0.3, 0.3, (None, None), [fit], np.zeros(1), np.zeros(1), 0.0, "test")
    assert with_catalogue(est, [on], {130: record}).books[0].call_number == 130
    assert with_catalogue(est, [on], {}).books[0] is fit                   # not in the catalogue: the camera's fit
