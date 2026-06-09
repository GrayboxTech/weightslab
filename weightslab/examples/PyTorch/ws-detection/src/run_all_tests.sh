#!/bin/bash
# Sequential scenario runs: discard → tag-discard → discard-all-val.
# Each test waits for trainer ramp on its own and exits 0/!=0 PASS/FAIL.
set -u
PY=/home/rotaru/anaconda3/envs/wl_13_yolo/bin/python
cd /home/rotaru/Desktop/GRAYBOX/onboard_exp_13_yolo/weightslab/weightslab/examples/PyTorch/ws-detection/src

for t in client_discard_test client_tag_discard_test client_discard_all_val_test; do
    echo "=== START: $t ==="
    date
    rm -f /tmp/${t}.log
    $PY -u $t.py > /tmp/${t}.log 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== PASS: $t ==="
    else
        echo "=== FAIL ($rc): $t ==="
        tail -20 /tmp/${t}.log
        exit $rc
    fi
done
echo "=== ALL PASS ==="
