# Paper experiments. Resumable: rerun to continue (rows already in an --out are skipped).
# Same setting for every row: real books (catalogue), 4 page sections, covers drawn, seeds 0-49 per scenario.
# Reruns write to results/slide_check/. The rows the report's tables come from are in results/paper/
# (read by notebooks/08_paper_draft.ipynb). The segmenter runs on CUDA, MPS or the CPU, whichever there is.
W=bila_weights/segmenter/segmenter.pt
out=results/slide_check
common="--catalogue --book-pages 4 --book-meshes --scenarios clear tight leaning packed --seeds 0:50"
ev="uv run --group rl python -m bila.control.eval_reshelve run $common"
mkdir -p $out

# 1. Camera with fiducial correction (Table 1) and without it (Table 2's real books), learned segmenter.
$ev --perception camera --segmenter $W --workers 3 --out $out/camera.jsonl > $out/camera.log 2>&1 &
$ev --perception camera --segmenter $W --no-fiducials --workers 4 \
    --out $out/camera_nofid.jsonl > $out/camera_nofid.log 2>&1 &

# 2. Truth column (Table 1), then the residual's held-out test: layouts from seed 30000, never trained or
#    selected on. The test needs runs/insertion_mix, from results/find_hard.sh and results/train_mix.sh.
(set -e
 $ev --perception truth --workers 2 --out $out/truth.jsonl
 for s in clear tight leaning; do
   [ -f scenes/gap_pool_${s}_test.pkl ] || uv run --group rl python -m bila.rl.gap_env pool --scenario $s \
       --seeds 120 --first-seed 30000 --perception camera --segmenter $W --name ${s}_test --workers 2
 done
 uv run --group rl python -m bila.rl.hard_cases label --pools clear_test tight_test leaning_test --segmenter $W \
     --workers 2 --out $out/labels_test.jsonl
 uv run --group rl python -m bila.rl.hard_cases mix --labels $out/labels_test.jsonl \
     --pools clear=clear_test tight=tight_test leaning=leaning_test --suffix _test50 --val-fraction 0
 if [ -d runs/insertion_mix ]; then
   uv run --group rl python -m bila.rl.hard_cases test --run runs/insertion_mix --workers 2 \
       --out $out/residual_test.json
 else
   echo "runs/insertion_mix not found: train it with results/find_hard.sh and results/train_mix.sh first"
 fi
) > $out/truth_residual.log 2>&1 &
wait
