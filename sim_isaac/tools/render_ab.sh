#!/usr/bin/env bash
# A/B of RTX presets for the head camera: render time and a sample frame per preset (no DDS).
#   bash sim_isaac/tools/render_ab.sh [HOUSE]
set -uo pipefail
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh
cd "$(dirname "$0")/../.."
HOUSE=${1:-procthor-train-40}
OUT=/work/worldline-g1/outputs/m1/isaac/render_ab
mkdir -p "$OUT"
PY=/work/envs/isaaclab/bin/python
for cfg in "balanced:--rendering_mode balanced" "performance:--rendering_mode performance" \
           "performance_dn:--rendering_mode performance --dl-denoiser" "balanced_dn:--rendering_mode balanced --dl-denoiser"; do
  name=${cfg%%:*}; args=${cfg#*:}
  PYTHONUNBUFFERED=1 $PY -m sim_isaac.app --house "$HOUSE" --no-dds --port-offset 100 --camera 640x480 --camera-hz 30 \
     --duration 25 --stats-out "$OUT/$name.json" --out-dir "$OUT/$name" $args > "$OUT/$name.log" 2>&1 &
  app=$!
  for _ in $(seq 1 200); do grep -q -E "WL_ISAAC_READY|Traceback" "$OUT/$name.log" && break; sleep 1; done
  sleep 5
  $PY - "$OUT/$name.png" <<'PYEOF'
import sys, base64, zmq, msgpack, numpy as np, cv2
s = zmq.Context.instance().socket(zmq.SUB); s.setsockopt(zmq.SUBSCRIBE, b""); s.connect("tcp://127.0.0.1:5665")
for _ in range(20):
    m = msgpack.unpackb(s.recv(), raw=False)
img = cv2.imdecode(np.frombuffer(base64.b64decode(m["images"]["ego_view"]), np.uint8), cv2.IMREAD_COLOR)
cv2.imwrite(sys.argv[1], img[..., ::-1])
g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
print("noise(laplacian std):", float(cv2.Laplacian(g, cv2.CV_32F).std()))
PYEOF
  wait $app
  python3 -c "import json; s=json.load(open('$OUT/$name.json')); print('$name', 'render_ms', s['render_ms']['mean'], s['render_ms']['p99'], 'rtf_10s', s['rtf_10s'])"
done
