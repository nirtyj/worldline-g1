# Shared pins and paths for the SONIC deploy component (P2 wl-sonic). Sourced by build_deploy.sh,
# run_deploy.sh and mujoco_ref/*.sh. Everything here is box-side (/work/...).
# shellcheck shell=bash

# Non-interactive ssh does not load the box profile (HF_HOME, TMPDIR, ...).
[[ -f /etc/profile.d/ludo.sh ]] && source /etc/profile.d/ludo.sh
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

SONIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WL_ROOT="$(cd "$SONIC_DIR/.." && pwd)"

# --- upstream checkout (pinned, shared, read-only for us apart from build/ target/ .venv_sim) ------
WBC_DIR="${WBC_DIR:-/work/repos/GR00T-WholeBodyControl}"
WBC_SHA=b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
DEPLOY_DIR="$WBC_DIR/gear_sonic_deploy"
DEPLOY_BIN="$DEPLOY_DIR/target/release/g1_deploy_onnx_ref"
VENV_SIM="$WBC_DIR/.venv_sim"

# --- toolchain (installed by build_deploy.sh) ------------------------------------------------------
# TensorRT must be EXACTLY 10.13 on x86_64 (WBC docs/source/getting_started/installation_deploy.md:16-24).
# The cuda-12.9 TAR is used because the deploy's reference environment is CUDA 12.x
# (installation_deploy.md:120-121: "x86_64: CUDA 12.4.1"); driver 580 runs CUDA 12.9 user space fine.
TRT_VERSION=10.13.3.9
TRT_TAR=TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz
TRT_URL=https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.13.3/tars/$TRT_TAR
TRT_BYTES=6933500463
DOWNLOADS=/work/downloads
OPT=/work/opt
export TensorRT_ROOT="$OPT/TensorRT-$TRT_VERSION"
# CUDA 12.9 runtime + headers + nvcc, extracted from the NVIDIA apt repo debs into /work/opt (NOT
# installed system-wide: the Nebius image pins o=NVIDIA to priority -1 and ships CUDA 13.0 only).
CUDA_VER=12.9
CUDA_PKG_SUFFIX=12-9
export CUDAToolkit_ROOT="$OPT/cuda-$CUDA_VER"
ORT_VERSION=1.16.3                 # scripts/install_deps.sh:383
ORT_DIR=/opt/onnxruntime           # install_deps.sh:380 default; scripts/setup_env.sh:24-30 looks here
JUST_VERSION=1.43.0                # install_deps.sh:339

# Runtime library path for the deploy binary (what scripts/setup_env.sh:176-306 would export).
deploy_ld_path() { echo "$TensorRT_ROOT/lib:$CUDAToolkit_ROOT/lib64:$ORT_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; }

# --- runtime defaults -------------------------------------------------------------------------------
# Build phase: our own tests use the contract ports +100 (see docs/contracts/sonic_deploy.md).
SONIC_ZMQ_PORT_DEFAULT=5556        # deploy SUBscribes (connects) to tcp://<host>:<port> (command/planner/pose)
SONIC_ZMQ_OUT_PORT_DEFAULT=5557    # deploy PUB binds tcp://*:<port>, topics g1_debug + robot_config
LOG_ROOT=/work/logs/wl
OUT_ROOT=/work/worldline-g1/outputs/m1/deploy
mkdir -p "$LOG_ROOT" 2>/dev/null || true

# PIDs of g1_deploy_onnx_ref processes that share OUR network namespace. DDS domain 0 is hard-coded in the binary
# (g1_deploy_onnx_ref.cpp:2218), so two deploys on the same `lo` would cross-talk; a deploy inside another netns
# (sim_isaac/tools/sonic_netns_test.sh, scripts/body_netns_e2e.sh) has its own `lo` and does not conflict.
# Matched on the executable (/proc/PID/exe), not the command line: `pgrep -f` alone also matches any shell whose
# command line merely mentions the binary name (e.g. an ssh one-liner that launches or greps for it).
deploys_in_my_netns() {
  local me p; me=$(readlink /proc/self/ns/net)
  for p in $(pgrep -f 'g1_deploy_onnx_ref' || true); do
    [[ "$(readlink "/proc/$p/exe" 2>/dev/null)" == */g1_deploy_onnx_ref ]] || continue
    [[ "$(readlink "/proc/$p/ns/net" 2>/dev/null)" == "$me" ]] && echo "$p"
  done
  return 0
}

# PIDs of running Isaac Lab / Isaac Sim python processes (argv[0] is the isaaclab env python). Matched on argv[0],
# not the whole command line, so shells that merely mention the path are not counted.
isaac_jobs() {
  local p a0
  for p in $(pgrep -f 'isaaclab/bin/python' || true); do
    a0=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)
    [[ "$a0" == */isaaclab/bin/python* ]] && echo "$p"
  done
  return 0
}

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { log "ERROR: $*"; exit 1; }
