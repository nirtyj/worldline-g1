#!/usr/bin/env bash
# Build the UNMODIFIED C++ gear_sonic_deploy (g1_deploy_onnx_ref) on the Brev box. Idempotent: every
# step checks its own result first and is skipped when already done. Run on the box:
#
#   bash /work/worldline-g1/sonic/build_deploy.sh            # everything
#   FORCE_BUILD=1 bash .../build_deploy.sh                   # clean C++ rebuild
#   SKIP_SIM_VENV=1 bash .../build_deploy.sh                 # skip the MuJoCo reference venv
#   STEPS="1 2 3" bash .../build_deploy.sh                   # run only these steps (default: all)
#
# Steps (upstream reference in brackets, WBC @ b042411):
#   1 apt C++ deps                     [gear_sonic_deploy/scripts/install_deps.sh:146-201]
#   2 just 1.43.0 + ONNX Runtime 1.16.3 -> /opt/onnxruntime   [install_deps.sh:333-441]
#   3 CUDA 12.9 cudart/headers/nvcc -> /work/opt/cuda-12.9 (extracted debs, no system install)
#   4 TensorRT 10.13.3.9 TAR -> /work/opt/TensorRT-10.13.3.9, ~/TensorRT symlink
#                                      [docs/source/getting_started/installation_deploy.md:16-40]
#   5 deploy ONNX + planner            [download_from_hf.py default flags: policy/release + planner V2]
#   6 cmake build of g1_deploy_onnx_ref [.justfile `build` recipe; deploy.sh:512-520 sources setup_env.sh first]
#   7 .venv_sim for the MuJoCo reference loop [install_scripts/install_mujoco_sim.sh]
#   8 smoke test: binary resolves its libs and prints usage
set -euo pipefail
source "$(dirname "$0")/sonic_env.sh"
LOG="$LOG_ROOT/deploy-build.log"
exec > >(tee -a "$LOG") 2>&1
T0=$(date +%s)
log "=== build_deploy.sh $(date -Is) on $(hostname) ==="
want() { [[ -z "${STEPS:-}" || " $STEPS " == *" $1 "* ]]; }
APT_LOCK=/var/lib/ludo-apt.lock   # shared with the other labs/agents on this box
apt_install() { sudo flock "$APT_LOCK" env DEBIAN_FRONTEND=noninteractive apt-get install -y -q "$@"; }

# ---------------------------------------------------------------------------------------------------
log "step 0: preflight"
[[ -d "$WBC_DIR/.git" ]] || die "$WBC_DIR missing"
sha=$(git -C "$WBC_DIR" rev-parse HEAD)
[[ "$sha" == "$WBC_SHA" ]] || die "WBC at $sha, expected $WBC_SHA (pinned). Not touching it."
head -c 8 "$DEPLOY_DIR/thirdparty/unitree_sdk2/lib/x86_64/libunitree_sdk2.a" | grep -q '!<arch>' \
  || die "git LFS objects missing in $WBC_DIR (git -C $WBC_DIR lfs pull)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || die "no GPU"

# ---------------------------------------------------------------------------------------------------
if want 1; then
log "step 1: apt packages"
# install_deps.sh apt list, Ubuntu 24.04 names: libmsgpack-dev pulls libmsgpack-cxx-dev (msgpack.hpp,
# needed by src/g1/g1_deploy_onnx_ref/CMakeLists.txt:40-50); cppzmq-dev provides /usr/include/zmq.hpp
# (install_deps.sh:225-278 vendors it otherwise). xdotool/pv are convenience only.
PKGS=(build-essential clang cmake git git-lfs pkg-config patchelf zlib1g-dev curl wget lsb-release
      libyaml-cpp-dev libeigen3-dev libmsgpack-dev libzmq3-dev cppzmq-dev nlohmann-json3-dev libgtest-dev
      ffmpeg xvfb pv)
missing=()
for p in "${PKGS[@]}"; do dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q "install ok installed" || missing+=("$p"); done
if ((${#missing[@]})); then
  log "installing: ${missing[*]}"
  apt_install "${missing[@]}" || { sudo flock "$APT_LOCK" apt-get update -q; apt_install "${missing[@]}"; }
else
  log "SKIP: all ${#PKGS[@]} apt packages present"
fi
[[ -f /usr/include/zmq.hpp && -f /usr/include/nlohmann/json.hpp && -f /usr/include/msgpack.hpp ]] \
  || die "cppzmq / nlohmann / msgpack headers missing after apt"
fi
# ---------------------------------------------------------------------------------------------------
if want 2; then
log "step 2: just $JUST_VERSION + ONNX Runtime $ORT_VERSION"
if command -v just >/dev/null && [[ "$(just --version)" == *"$JUST_VERSION"* ]]; then
  log "SKIP: $(just --version)"
else
  t=$(mktemp -d)
  curl -fsSL "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" | tar xz -C "$t"
  sudo install -m 0755 "$t/just" /usr/local/bin/just; rm -rf "$t"
  log "installed $(just --version)"
fi
if [[ -e "$ORT_DIR/lib/libonnxruntime.so.$ORT_VERSION" ]]; then
  log "SKIP: onnxruntime $ORT_VERSION at $ORT_DIR"
else
  [[ -e "$ORT_DIR" ]] && die "$ORT_DIR exists but is not onnxruntime $ORT_VERSION; move it away first"
  t=$(mktemp -d)
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" | tar xz -C "$t"
  sudo mv "$t/onnxruntime-linux-x64-${ORT_VERSION}" "$ORT_DIR"; rm -rf "$t"
  # same links + ld.so.conf entry as install_deps.sh:414-420
  sudo ln -sf "$ORT_DIR/lib/libonnxruntime.so" /usr/local/lib/
  sudo ln -sfn "$ORT_DIR/include" /usr/local/include/onnxruntime
  echo "$ORT_DIR/lib" | sudo tee /etc/ld.so.conf.d/onnxruntime.conf >/dev/null
  sudo ldconfig
  log "installed onnxruntime $ORT_VERSION -> $ORT_DIR"
fi
# gear_sonic_deploy/cmake/Findonnxruntime.cmake:10-30 searches /opt/onnxruntime/{include,lib} directly.
fi
# ---------------------------------------------------------------------------------------------------
if want 3; then
log "step 3: CUDA $CUDA_VER runtime/headers/nvcc -> $CUDAToolkit_ROOT"
cuda_ok() {
  [[ -x "$CUDAToolkit_ROOT/bin/nvcc" && -e "$CUDAToolkit_ROOT/lib64/libcudart.so.12" \
     && -f "$CUDAToolkit_ROOT/include/cuda_runtime.h" && -f "$CUDAToolkit_ROOT/include/crt/host_config.h" \
     && -f "$CUDAToolkit_ROOT/include/cuda.h" ]]
}
if cuda_ok; then
  log "SKIP: $("$CUDAToolkit_ROOT/bin/nvcc" --version | tail -2 | head -1)"
else
  t=$(mktemp -d); mkdir -p "$t/root"
  for p in cudart cudart-dev cccl driver-dev nvcc nvvm crt; do
    pkg="cuda-$p-$CUDA_PKG_SUFFIX"
    ver=$(apt-cache madison "$pkg" | awk -F'|' 'NR==1{gsub(/ /,"",$2); print $2}')
    [[ -n "$ver" ]] || die "no apt candidate for $pkg"
    (cd "$t" && apt-get download -q "$pkg=$ver") || die "apt-get download $pkg=$ver failed"
  done
  for d in "$t"/*.deb; do dpkg -x "$d" "$t/root"; done
  mkdir -p "$OPT"; rm -rf "$CUDAToolkit_ROOT"
  mv "$t/root/usr/local/cuda-$CUDA_VER" "$CUDAToolkit_ROOT"; rm -rf "$t"
  cuda_ok || die "CUDA extraction incomplete in $CUDAToolkit_ROOT"
  log "extracted $(ls "$CUDAToolkit_ROOT/lib64/" | grep -m1 'libcudart.so.12\.') + nvcc to $CUDAToolkit_ROOT"
fi
fi
# ---------------------------------------------------------------------------------------------------
if want 4; then
log "step 4: TensorRT $TRT_VERSION"
trt_ok() { [[ -e "$TensorRT_ROOT/lib/libnvinfer.so.10" && -f "$TensorRT_ROOT/include/NvInferVersion.h" ]]; }
if trt_ok; then
  log "SKIP: TensorRT at $TensorRT_ROOT"
else
  mkdir -p "$DOWNLOADS/tensorrt" "$OPT"
  tb="$DOWNLOADS/tensorrt/$TRT_TAR"
  size() { stat -c %s "$1" 2>/dev/null || echo 0; }
  # Direct NVIDIA URL, no login (HTTP 200 checked 2026-09-28). Downloading means accepting the TensorRT SLA.
  [[ "$(size "$tb")" == "$TRT_BYTES" ]] || curl -fL --retry 5 --retry-delay 5 -C - -o "$tb" "$TRT_URL"
  [[ "$(size "$tb")" == "$TRT_BYTES" ]] || die "TensorRT tarball size $(size "$tb") != $TRT_BYTES"
  log "extracting $tb"
  tar -xzf "$tb" -C "$OPT"
  trt_ok || die "extraction did not produce $TensorRT_ROOT/lib/libnvinfer.so.10"
fi
# deploy.sh:423-435 falls back to ~/TensorRT when TensorRT_ROOT is unset.
if [[ -L "$HOME/TensorRT" || ! -e "$HOME/TensorRT" ]]; then ln -sfn "$TensorRT_ROOT" "$HOME/TensorRT"; fi
trt_hdr=$(awk '/#define TRT_(MAJOR|MINOR|PATCH|BUILD)_ENTERPRISE /{printf "%s.", $3}' "$TensorRT_ROOT/include/NvInferVersion.h"); trt_hdr=${trt_hdr%.}
[[ "$trt_hdr" == "$TRT_VERSION" ]] || die "TensorRT headers report '$trt_hdr', expected $TRT_VERSION"
log "TensorRT headers report $trt_hdr"
fi
# ---------------------------------------------------------------------------------------------------
if want 5; then
log "step 5: deploy ONNX + planner (download_from_hf.py, default flags)"
onnx_ok() {
  [[ -s "$DEPLOY_DIR/policy/release/model_encoder.onnx" && -s "$DEPLOY_DIR/policy/release/model_decoder.onnx" \
     && -s "$DEPLOY_DIR/policy/release/observation_config.yaml" && -s "$DEPLOY_DIR/planner/target_vel/V2/planner_sonic.onnx" ]]
}
if onnx_ok; then
  log "SKIP: policy/release + planner/target_vel/V2 present"
else
  # download_from_hf.py:30-51 (POLICY_FILES, PLANNER_FILE); writes under gear_sonic_deploy/ by default.
  (cd "$WBC_DIR" && uv run --no-project --python 3.10 --with huggingface_hub python download_from_hf.py) \
    || die "download_from_hf.py failed"
  onnx_ok || die "deploy ONNX still missing"
fi
ls -la "$DEPLOY_DIR/policy/release" "$DEPLOY_DIR/planner/target_vel/V2" | sed 's/^/  /'
fi
# ---------------------------------------------------------------------------------------------------
if want 6; then
log "step 6: C++ build (g1_deploy_onnx_ref)"
(
  cd "$DEPLOY_DIR"
  set +eu
  export HAS_ROS2=0          # no ROS2 on the box; src/g1/g1_deploy_onnx_ref/CMakeLists.txt:1-24
  # deploy.sh:512-516 sources this before `just build`; it sets onnxruntime_DIR, CMAKE_PREFIX_PATH and the
  # CUDA/TRT library paths from CUDAToolkit_ROOT + TensorRT_ROOT (setup_env.sh:165-306).
  source scripts/setup_env.sh >/dev/null
  set -eu
  [[ "${FORCE_BUILD:-0}" == 1 ]] && rm -rf build target
  mkdir -p build
  # == the .justfile `build` recipe (cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release ...), but only the
  # deploy target and -j8 so a build does not take all 16 vCPUs from the other agents on the box.
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCUDAToolkit_ROOT="$CUDAToolkit_ROOT" 2>&1 | grep -E "CUDA|TensorRT|onnx|ZMQ|msgpack|ROS2|Error|error|WARN" || true
  cmake --build build -j"${BUILD_JOBS:-8}" --target g1_deploy_onnx_ref
)
[[ -x "$DEPLOY_BIN" ]] || die "build finished but $DEPLOY_BIN is missing"
fi
# ---------------------------------------------------------------------------------------------------
if want 7; then
if [[ "${SKIP_SIM_VENV:-0}" == 1 ]]; then
  log "step 7: SKIP .venv_sim (SKIP_SIM_VENV=1)"
else
  log "step 7: .venv_sim for the MuJoCo reference loop"
  sim_ok() { [[ -x "$VENV_SIM/bin/python" ]] && "$VENV_SIM/bin/python" -c 'import mujoco, unitree_sdk2py, zmq, tyro, gear_sonic, msgpack' 2>/dev/null; }
  if sim_ok; then
    log "SKIP: .venv_sim (mujoco $("$VENV_SIM/bin/python" -c 'import mujoco; print(mujoco.__version__)'))"
  else
    (cd "$WBC_DIR" && bash install_scripts/install_mujoco_sim.sh) || die "install_mujoco_sim.sh failed"
    sim_ok || die ".venv_sim built but imports fail"
  fi
fi
fi
# ---------------------------------------------------------------------------------------------------
if want 8; then
log "step 8: smoke test"
# trt_hdr is set by step 4; recompute it when step 8 runs alone (STEPS="8") so `set -u` does not abort
trt_hdr=${trt_hdr:-$(awk '/#define TRT_(MAJOR|MINOR|PATCH|BUILD)_ENTERPRISE /{printf "%s.", $3}' "$TensorRT_ROOT/include/NvInferVersion.h" 2>/dev/null)}; trt_hdr=${trt_hdr%.}
# argc < 4 prints usage and exits before any CUDA/DDS call (g1_deploy_onnx_ref.cpp:4147-4205).
out=$(cd "$DEPLOY_DIR" && LD_LIBRARY_PATH="$(deploy_ld_path)" "$DEPLOY_BIN" 2>&1 | head -3 || true)
echo "$out" | grep -q "Usage:" || die "deploy binary does not start: $out"
(cd "$DEPLOY_DIR" && LD_LIBRARY_PATH="$(deploy_ld_path)" ldd "$DEPLOY_BIN") | grep -E "nvinfer|onnxruntime|cudart|zmq|not found" | sed 's/^/  /'
if (cd "$DEPLOY_DIR" && LD_LIBRARY_PATH="$(deploy_ld_path)" ldd "$DEPLOY_BIN") | grep -q "not found"; then die "unresolved libraries"; fi
log "OK: $DEPLOY_BIN ($(du -h "$DEPLOY_BIN" | cut -f1)), TensorRT $trt_hdr, CUDA $CUDA_VER, onnxruntime $ORT_VERSION, $(( $(date +%s) - T0 ))s"
fi
