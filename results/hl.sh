set -e
# The learned skill selector: episodes of the rules with random deviations (seeds 5000-5059), then the
# Q-function fitted on them (runs/hl/q.npz), which results/hl_eval.sh and results/hl_cart.sh use.
mkdir -p results/slide_check
uv run python -m bila.control.high_level collect --perception noisy-learned --seeds 5000:5060 --eps 0.3 \
    --out results/slide_check/hl_train.jsonl --workers 4
uv run --group rl python -m bila.control.high_level fit --data results/slide_check/hl_train.jsonl --out runs/hl/q.npz
