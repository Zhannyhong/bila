uv run --group rl python -m bila.control.eval_reshelve run --perception camera --segmenter ~/bila_data/segmenter/segmenter.pt \
    --segmenter-device mps --catalogue --book-pages 4 --book-meshes --scenarios clear tight leaning packed --seeds 0:50 \
    --workers 5 --out results/slide_check/camera.jsonl >> results/slide_check/camera.log 2>&1
