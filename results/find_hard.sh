set -e
W=~/bila_data/segmenter/segmenter.pt
for s in clear tight leaning; do
  [ -f scenes/gap_pool_${s}_cand.pkl ] || uv run --group rl python -m bila.rl.gap_env pool --scenario $s --seeds 250 \
      --first-seed 10000 --perception camera --segmenter $W --segmenter-device cpu --name ${s}_cand --workers 12
done
uv run --group rl python -m bila.rl.hard_cases label --pools clear_cand tight_cand leaning_cand --segmenter $W \
    --segmenter-device cpu --workers 12 --out results/labels.jsonl
