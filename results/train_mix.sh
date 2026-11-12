uv run --group rl python -m bila.rl.train_residual --name insertion_mix --scenarios clear tight leaning \
  --train-suffix _mix --val-suffix _mix_val --val-scenes 100 --control-phases slide push --no-learn-stop \
  --perception camera --segmenter ~/bila_data/segmenter/segmenter.pt --segmenter-device cpu \
  --workers 10 --decisions 150000 --ramp-decisions 50000 --eval-every 5
