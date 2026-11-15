set -e
uv run python -m bila.control.high_level collect --perception noisy-learned --seeds 5000:5060 --eps 0.3 \
    --out results/hl_train.jsonl --workers 4
