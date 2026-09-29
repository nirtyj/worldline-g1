#!/usr/bin/env bash
# Start the Arena base container (isaaclab_arena:latest built from release/0.2.1 docker/Dockerfile.isaaclab_arena)
# detached, with the same mounts/flags as docker/run_docker.sh, plus /spike (our policies) and /eval -> outputs.
set -euo pipefail
A=/work/arena
sudo docker rm -f arena >/dev/null 2>&1 || true
mkdir -p $A/eval $A/cache/home
sudo docker run -d --name arena \
  --privileged --ulimit memlock=-1 --ulimit stack=-1 --ipc=host --net=host --runtime=nvidia --gpus=all \
  -v $A/IsaacLab-Arena:/workspaces/isaaclab_arena \
  -v $A/datasets:/datasets -v $A/models:/models -v $A/eval:/eval -v $A/spike:/spike \
  -v $A/cache/home:/home/ubuntu/.cache \
  -v /etc/ssl/certs:/etc/ssl/certs:ro \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES \
  --env DOCKER_RUN_USER_ID=$(id -u) --env DOCKER_RUN_USER_NAME=$(id -un) \
  --env DOCKER_RUN_GROUP_ID=$(id -g) --env DOCKER_RUN_GROUP_NAME=$(id -gn) \
  --env ISAACLAB_PATH=/workspaces/isaaclab_arena/submodules/IsaacLab \
  --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  isaaclab_arena:latest "sleep infinity"
sleep 5
sudo docker ps --filter name=arena
