# the cart demo (wrist camera, fiducials, simulation's masks), the rules and the policy on the same scenes
out=results/hl_cart.txt; : > $out
for s in clear tight leaning packed; do for seed in 0 1 2; do for mode in rules policy; do
  extra=""; [ $mode = policy ] && extra="--policy runs/hl/q.npz"
  line=$(uv run python -m bila.control.cart_demo --scenario $s --seed $seed $extra 2>&1 | grep -E "SUCCESS|FAIL|skipped|MISSED|  [A-Z]+ \[" | tail -1)
  echo "$s $seed $mode | $line" >> $out
done; done; done
