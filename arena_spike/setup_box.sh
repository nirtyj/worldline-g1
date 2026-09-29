#!/usr/bin/env bash
# Arena spike: install IsaacLab-Arena (release/0.2.1, Isaac Sim 6.0.0-dev2 / Isaac Lab 3.0) + two GR00T envs
# on the dedicated spike box (ludo-g1-arena). Idempotent; run in tmux:  bash setup_box.sh 2>&1 | tee /work/arena/logs/setup.log
set -euo pipefail
source /etc/profile.d/ludo.sh
A=/work/arena
mkdir -p $A/{outputs,models,datasets,logs,eval} /scratch/tmp
GH="-c url.https://github.com/.insteadOf=git@github.com:"

ARENA_REF=release/0.2.1            # only branch that ships galileo_g1_static_pick_and_place (N1.7 ckpt)
GR00T_N17_SHA=4b1dca9d88d2a0b9ea5a65aa61c82ff89f5c4f0e   # static_apple docs pin (standalone N1.7 server)
# N1.6 server = Arena's own submodule submodules/Isaac-GR00T @ e29d8fc (locomanip docs)

echo "== [1] clone Arena $ARENA_REF"
if [ ! -d $A/IsaacLab-Arena/.git ]; then
  git clone -b $ARENA_REF https://github.com/isaac-sim/IsaacLab-Arena.git $A/IsaacLab-Arena
fi
cd $A/IsaacLab-Arena
git $GH submodule update --init
git log -1 --format='arena %H %ci %s'
git submodule status

echo "== [2] standalone Isaac-GR00T for N1.7"
if [ ! -d $A/gr00t_n17/.git ]; then
  git clone https://github.com/NVIDIA/Isaac-GR00T.git $A/gr00t_n17
fi
cd $A/gr00t_n17 && git fetch -q origin && git checkout -q $GR00T_N17_SHA && git $GH submodule update --init
git log -1 --format='gr00t_n17 %H %ci %s'

echo "== [3] docker base image + Arena image"
sudo docker pull nvcr.io/nvidia/isaac-sim:6.0.0-dev2
cd $A/IsaacLab-Arena
sudo docker build --progress=plain \
  --build-arg WORKDIR=/workspaces/isaaclab_arena --build-arg INSTALL_GROOT=false \
  -t isaaclab_arena:latest -f docker/Dockerfile.isaaclab_arena .
echo "== setup done"
