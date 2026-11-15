"""The high-level policy's bookkeeping: valid actions, rewards, the masked argmax."""

import numpy as np

from bila.control import high_level as hl

A = hl.ACTIONS.index


def feats(**kw):
    f = dict.fromkeys(hl.FEATURES, 0.0)
    f.update(gap_spare_mm=10.0, widen_spare_mm=4.0, fingers_spare_mm=20.0, last_did=1.0, **kw)
    return np.array([f[k] for k in hl.FEATURES])


def test_valid_actions():
    m = hl.valid(feats(), [], 0, False)
    assert m[A("insert")] and m[A("widen_more")] and m[A("give_up")]
    assert not m[A("widen")]                                   # wide enough: the plain widening has nothing to do
    assert not m[A("set_aside")] and not m[A("stand_up_left")]
    held = hl.valid(feats(), [("insert", False)], 1, True)      # a jammed book in the hand
    assert set(np.flatnonzero(held)) == {A("set_aside"), A("give_up")}
    capped = hl.valid(feats(), [("widen", True)] * hl.MAX_STEPS, 0, False)
    assert set(np.flatnonzero(capped)) == {A("give_up")}


def test_rewards_count_only_the_last_step_of_a_success():
    rows = [{"success": True, "steps": [(feats().tolist(), [True] * 8, A("widen")), (feats().tolist(), [True] * 8, A("insert"))]},
            {"success": False, "steps": [(feats().tolist(), [True] * 8, A("insert"))]}]
    S, a, R, S2, M2, D = hl.transitions(rows)
    assert np.allclose(R, [-hl.STEP_COST, 1 - hl.STEP_COST, -hl.STEP_COST]) and D.tolist() == [False, True, True]
    assert not M2[1].any() and M2[0].all()


def test_qpolicy_picks_the_best_valid_action(tmp_path):
    n = len(hl.FEATURES)
    W = np.zeros((n, len(hl.ACTIONS)))
    b = np.arange(len(hl.ACTIONS), dtype=float)                # give_up (last) best, then set_aside
    np.savez(tmp_path / "q.npz", mean=np.zeros(n), std=np.ones(n), n=1, W0=W, b0=b)
    q = hl.QPolicy(tmp_path / "q.npz")
    mask = np.zeros(len(hl.ACTIONS), bool)
    mask[[A("insert"), A("widen_more")]] = True
    assert q(None, feats(), mask, []) == A("widen_more")
