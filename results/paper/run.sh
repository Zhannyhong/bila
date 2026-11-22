# Paper experiments. Resumable: rerun to continue (rows already in an --out are skipped).
# Same setting for every row: real books (catalogue), 4 page sections, covers drawn, seeds 0-49 per scenario.
W=bila_weights/segmenter/segmenter.pt
common="--catalogue --book-pages 4 --book-meshes --scenarios clear tight leaning packed --seeds 0:50"
ev="uv run --group rl python -m bila.control.eval_reshelve run $common"

# 1. Perception ablation: truth / camera + fiducial correction / camera without it (learned segmenter).
$ev --perception camera --segmenter $W --segmenter-device mps --workers 3 \
    --out results/slide_check/camera.jsonl > results/paper/camera.log 2>&1 &
$ev --perception camera --segmenter $W --segmenter-device mps --no-fiducials --workers 4 \
    --out results/paper/camera_nofid.jsonl > results/paper/camera_nofid.log 2>&1 &

# 2. Error source: the truth plus the camera's measured errors (learned-mask pipeline), scaled; no rendering.
(for s in 1 0.5 2; do
   $ev --perception noisy-learned --noise-scale $s --workers 3 --out results/paper/noisy_x$s.jsonl
 done) > results/paper/noisy.log 2>&1 &

# 3. Truth column, then the residual's held-out test (layouts from seed 30000, never trained or selected on).
(set -e
 $ev --perception truth --workers 2 --out results/slide_check/truth.jsonl
 uv run --group rl python -m bila.rl.hard_cases label --pools clear_test tight_test leaning_test --segmenter $W \
     --segmenter-device mps --workers 2 --out results/labels_test.jsonl
 uv run --group rl python -m bila.rl.hard_cases mix --labels results/labels_test.jsonl \
     --pools clear=clear_test tight=tight_test leaning=leaning_test --suffix _test50 --val-fraction 0
 uv run --group rl python -m bila.rl.hard_cases test --run runs/insertion_mix --workers 2 --out results/residual_test.json
) > results/paper/truth_residual.log 2>&1 &
wait
