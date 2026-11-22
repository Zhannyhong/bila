"""Does the insertion grasp hold the book? Take it as insert_book does
(take_book), but in free air in front of the board, carry it 6 cm out and
3 cm up at transit speed and back, hold it 1 s, and measure how far it turned
and slid in the hand. The book's own friction is at the bottom of its range
(the pads' PAD_FRICTION wins anyway: a contact takes the larger).

Known ceiling: the pads pinch the book near its spine, so its weight acts
several cm behind them and turns it nose-down. A heavy, deep book creeps or is
dropped whatever the pad friction; insert_book gets away with it by resting
the book on the board straight away. Those cases are strict xfails: if the
grasp gets fixed, they start passing and have to be moved up.

The cart tests run the real carry (cart_demo: pick from the tray, swing it to
face the shelf, insert): the heaviest book allowed (MAX_KG) gets in, and on
weaker pads a 3 kg book trips the slip watch (CartEpisode.check_grip). The
cart's grip (PAD_TORSION, elliptic cones) is far stronger than the plain
scene's, which is why the free-air cases above fail and these pass.

    uv run pytest tests/test_grasp_slip.py
"""

import math

import mujoco
import numpy as np
import pytest

from bila.control import insert_book as ib
from bila.sim.shelf_scene import ShelfConfig, sample_row, settle

TURN_OK_DEG, SLIDE_OK_MM = 5.0, 5.0


def hold(thickness, depth, height, mass, seed=0):
    """(worst turn in the hand, deg; worst slide, mm) while carrying the book."""
    cfg = ShelfConfig()
    rng = np.random.default_rng(seed)
    for _ in range(30):                                   # a level tall enough to lift it in
        row = sample_row(cfg, "clear", rng)
        if row.clear_height > height + 0.10:
            break
    b = row.target_book
    b.thickness, b.depth, b.height, b.mass, b.friction = thickness, depth, height, mass, cfg.book_friction[0]
    m = ib.build_model(cfg, row, ib.DEFAULT_PANDA)
    d = mujoco.MjData(m)
    settle(m, d, row)
    ep = ib.Episode(m, d, row, log=lambda *a: None)
    pre = ep.book_pre(ep.channel()[1])
    pre[1] += 0.02                                        # its back 1 cm in front of the board: nothing under it
    cands = [(ep.grasp_pose(gs, pre), ep.insertion_plan(gs, 1, pre[0], pre[1], ep.obstacles()), gs) for gs in (1, -1)]
    ep.take_book(ep.begin(cands), pre)

    run, hand, book = ep.run, ep.ik.hand, ep.book_body

    def in_hand():
        Rh = d.xmat[hand].reshape(3, 3)
        return Rh.T @ (d.xpos[book] - d.xpos[hand]), Rh.T @ d.xmat[book].reshape(3, 3)

    p0, R0 = in_hand()
    worst = [0.0, 0.0]

    def track():
        p, R = in_hand()
        worst[0] = max(worst[0], math.degrees(math.acos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1))))
        worst[1] = max(worst[1], 1000 * np.linalg.norm(p - p0))

    start = run.pos.copy()
    for target in (start + ep.ys * 0.06 + [0, 0, 0.03], start):
        run.move("carry", target)
        track()
    run.wait("hold", 1.0)
    track()
    return tuple(worst)


ceiling = pytest.mark.xfail(strict=True, reason="heavy, deep book turns in the spine pinch (see module docstring)")


@pytest.mark.parametrize("thickness, depth, height, mass", [
    pytest.param(0.029, 0.153, 0.230, 0.828, id="typical book-ai86 0.83 kg"),
    pytest.param(0.038, 0.200, 0.275, 1.764, id="heaviest catalogue target book-schmidt 1.76 kg", marks=ceiling),
    pytest.param(0.060, 0.220, 0.300, 0.06 * 0.22 * 0.30 * 900, id="largest sampled book 3.56 kg", marks=ceiling),
])
def test_grasp_holds(thickness, depth, height, mass):
    turn, slide = hold(thickness, depth, height, mass)
    assert turn < TURN_OK_DEG and slide < SLIDE_OK_MM, f"turned {turn:.1f} deg, slid {slide:.1f} mm in the hand"



# --- The real carry: cart_demo picks the book out of the tray, swings it to face the shelf and inserts it.

def cart_insert(mass, torsion=None, seed=0):
    """cart_demo's clear row, the target book made `mass` kg, the pads' torsional
    friction `torsion` (None: PAD_TORSION); reshelve it from the tray."""
    from bila.control import cart_demo as cd
    model, data, row, _, numbers, shelf_map = cd.setup("clear", seed, log=lambda *a: None)
    body = model.body(row.target_book.name).id
    model.body_inertia[body] *= mass / model.body_mass[body]
    model.body_mass[body] = mass
    if torsion is not None:
        for g in range(model.ngeom):
            if model.body(model.geom_bodyid[g]).name in ("panda/left_finger", "panda/right_finger"):
                model.geom_friction[g, 1] = torsion
    _, top = cd.row_for(numbers[row.target_book.name], shelf_map)
    data.ctrl[model.actuator("lift").id] = top - cd.REACH_BELOW - cd.DECK
    return cd.CartEpisode(model, data, row, log=lambda *a: None).reshelve(ib.GAP_MARGIN)


def test_cart_carries_the_heaviest_allowed_book():
    from bila.control.cart_demo import MAX_KG
    assert cart_insert(MAX_KG)["success"]


def test_cart_stops_when_the_book_slips(monkeypatch):
    """3 kg on pads of torsion 0.01 (7 deg of creep on the way): the slip watch stops it. The watch
    reads the slip from the simulation, so it runs only with CartEpisode.PRIVILEGED."""
    from bila.control import cart_demo as cd
    monkeypatch.setattr(cd.CartEpisode, "PRIVILEGED", True)
    with pytest.raises(RuntimeError, match="slipped in the grip"):
        cart_insert(3.0, torsion=0.01)


def test_twist_gate_follows_the_pads(monkeypatch):
    from bila.control import cart_demo as cd
    assert not cd.shelvable(0.03, 0.20, 0.24, 1.4)                      # today's pads: fine
    monkeypatch.setattr(cd, "PAD_TORSION", 0.01)                        # realistic pads: too much twist
    assert "its weight would twist it in the grip" in cd.shelvable(0.03, 0.20, 0.24, 1.4)
