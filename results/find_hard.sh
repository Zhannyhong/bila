set -e
# The residual's training pools: candidate insertions (layouts from seed 10000) labelled by the controller
# alone, then mixed 50/50 into scenes/gap_pool_<scenario>_mix.pkl and _mix_val.pkl for results/train_mix.sh.
W=bila_weights/segmenter/segmenter.pt
mkdir -p results/slide_check
for s in clear tight leaning; do
  [ -f scenes/gap_pool_${s}_cand.pkl ] || uv run --group rl python -m bila.rl.gap_env pool --scenario $s --seeds 250 \
      --first-seed 10000 --perception camera --segmenter $W --segmenter-device cpu --name ${s}_cand --workers 12
done
uv run --group rl python -m bila.rl.hard_cases label --pools clear_cand tight_cand leaning_cand --segmenter $W \
    --segmenter-device cpu --workers 12 --out results/slide_check/labels.jsonl
ls scenes/gap_pool_*_mix.pkl > /dev/null 2>&1 || \
    uv run --group rl python -m bila.rl.hard_cases mix --labels results/slide_check/labels.jsonl --suffix _mix
