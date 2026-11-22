while pgrep -f "high_level collect" > /dev/null; do sleep 30; done
bash results/hl_cart.sh > results/hl_cart.log 2>&1 &
W=bila_weights/segmenter/segmenter.pt
for s in clear tight leaning; do
  [ -f scenes/gap_pool_${s}_test.pkl ] || uv run --group rl python -m bila.rl.gap_env pool --scenario $s --seeds 120 \
      --first-seed 30000 --perception camera --segmenter $W --segmenter-device cpu --name ${s}_test --workers 4
done
uv run --group rl python -m bila.rl.hard_cases label --pools clear_test tight_test leaning_test --segmenter $W \
    --segmenter-device cpu --workers 4 --out results/labels_test.jsonl
uv run --group rl python -m bila.rl.hard_cases mix --labels results/labels_test.jsonl \
    --pools clear=clear_test tight=tight_test leaning=leaning_test --suffix _test50 --val-fraction 0
wait
