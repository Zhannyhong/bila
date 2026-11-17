# The Results slide's columns, rerun locally: real books (catalogue), 4 page sections, covers drawn, seeds 0-49
common="--catalogue --book-pages 4 --book-meshes --scenarios clear tight leaning packed --seeds 0:50"
uv run python -m bila.control.eval_reshelve run --perception truth $common --workers 2 \
    --out results/slide_check/truth.jsonl > results/slide_check/truth.log 2>&1 &
uv run --group rl python -m bila.control.eval_reshelve run --perception camera --segmenter ~/bila_data/segmenter/segmenter.pt \
    --segmenter-device cpu $common --workers 5 --out results/slide_check/camera.jsonl > results/slide_check/camera.log 2>&1 &
wait
