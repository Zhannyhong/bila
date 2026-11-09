import numpy as np

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
