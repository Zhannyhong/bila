# The rules and the fitted selector (runs/hl/q.npz, from results/hl.sh) on the same 100 test rows.
mkdir -p results/slide_check
uv run python -m bila.control.high_level collect --perception noisy-learned --seeds 0:25 --eps 0 \
    --out results/slide_check/hl_eval_rules.jsonl --workers 2 > results/slide_check/hl_eval_rules.log 2>&1 &
uv run python -m bila.control.high_level collect --perception noisy-learned --seeds 0:25 --policy runs/hl/q.npz \
    --out results/slide_check/hl_eval_q.jsonl --workers 2 > results/slide_check/hl_eval_q.log 2>&1 &
wait
