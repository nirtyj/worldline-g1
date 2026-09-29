#!/usr/bin/env bash
# Two separate GR00T server envs (uv, python 3.10), one per checkpoint generation:
#   gr00t_n16 = Isaac-GR00T @ e29d8fc (== Arena release/0.2.1 submodule pin; N1.6 locomanip docs)
#   gr00t_n17 = Isaac-GR00T @ 4b1dca9 (static_apple docs pin; N1.7)
# The N1.6 env lives in its own clone so the Arena submodule stays clean for the docker COPY.
set -euo pipefail
source /etc/profile.d/ludo.sh
export CUDA_HOME=/usr/local/cuda   # flash-attn's setup.py needs a CUDA_HOME even when it downloads the prebuilt wheel
A=/work/arena
N16_SHA=e29d8fc50b0e4745120ae3fb72447986fe638aa6
N17_SHA=4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e
which="${1:-both}"

if [[ $which == both || $which == n16 ]]; then
  echo "== N1.6 env"
  [ -d $A/gr00t_n16/.git ] || git clone https://github.com/NVIDIA/Isaac-GR00T.git $A/gr00t_n16
  cd $A/gr00t_n16 && git fetch -q origin && git checkout -q $N16_SHA && git log -1 --format='gr00t_n16 %H %ci %s'
  uv sync --python 3.10
  uv pip install -e .
  .venv/bin/python -c "import torch, transformers, gr00t; import flash_attn; print('n16 OK torch', torch.__version__, torch.version.cuda, 'tf', transformers.__version__, 'fa', flash_attn.__version__, torch.cuda.is_available())"
fi
if [[ $which == both || $which == n17 ]]; then
  echo "== N1.7 env"
  [ -d $A/gr00t_n17/.git ] || git clone https://github.com/NVIDIA/Isaac-GR00T.git $A/gr00t_n17
  cd $A/gr00t_n17 && git fetch -q origin && git checkout -q $N17_SHA && git log -1 --format='gr00t_n17 %H %ci %s'
  uv sync --python 3.10
  uv pip install -e .
  .venv/bin/python -c "import torch, transformers, gr00t; import flash_attn; print('n17 OK torch', torch.__version__, torch.version.cuda, 'tf', transformers.__version__, 'fa', flash_attn.__version__, torch.cuda.is_available())"
fi
echo "== gr00t envs done"
