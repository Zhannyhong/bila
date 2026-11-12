"""mix: every failure and as many successes, each layout on one side of the split."""

import json
import pickle
from types import SimpleNamespace

import numpy as np

from bila.rl import gap_env, hard_cases


def test_mix_is_half_failures_and_keeps_layouts_apart(tmp_path, monkeypatch):
    outcomes = ["missed", "failure", "missed", "success", "success", "success", "success", "success",
                "preparation_failed", "truncated"]
    pool = [gap_env.Packed(gap_env.PoolEntry("tight", 100 + i, None, np.zeros(0), contact=True)) for i in range(10)]
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps({**p.unpack().identity(), "outcome": o}) + "\n" for p, o in zip(pool, outcomes)))
    monkeypatch.setattr(gap_env, "load_pool", lambda name, root=None: pool)
    monkeypatch.setattr(gap_env, "pool_path", lambda name, root=None: str(tmp_path / f"{name}.pkl"))

    hard_cases.mix(SimpleNamespace(labels=str(labels), pools=None, suffix="_mix", val_fraction=0.25, seed=0))
    split = {t: [p.seed for p in pickle.load(open(tmp_path / f"tight_mix{t}.pkl", "rb"))] for t in ("", "_val")}
    failed = {100 + i for i, o in enumerate(outcomes) if o in hard_cases.FAILED}
    both = split[""] + split["_val"]
    assert len(both) == len(set(both)) == 8                              # 4 failures, 4 of the 5 successes
    assert set(both) >= failed and 108 not in both                       # never a case the residual can't act on
    for seeds in split.values():
        assert 2 * sum(s in failed for s in seeds) == len(seeds)         # half failures on each side
    assert len(split["_val"]) == 2
