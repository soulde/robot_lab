# syntax=docker/dockerfile:1.7

FROM nvcr.io/nvidia/isaac-lab:3.0.0-rc1

USER root

ARG RSL_RL_COMMIT=8eef047f1d9b5e07d54d743387da2f0fb42791fa
ARG RSL_RL_BRANCH=develop
ARG ISAACLAB_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh

ENV ROBOT_LAB_ROOT=/workspace/robot_lab \
    ISAACLAB_PYTHON=${ISAACLAB_PYTHON} \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONUNBUFFERED=1

WORKDIR ${ROBOT_LAB_ROOT}
COPY . ${ROBOT_LAB_ROOT}

# The launcher exports this exact commit from the host's RSL-RL checkout as a
# small, clean BuildKit context. This avoids embedding credentials or a host
# virtual environment in the image build.
COPY --from=rsl_rl / /opt/robot_lab_rsl_rl

RUN test -f /opt/robot_lab_rsl_rl/setup.py && \
    test -x "${ISAACLAB_PYTHON}" && \
    ("${ISAACLAB_PYTHON}" -m pip uninstall -y rsl-rl-lib || true) && \
    "${ISAACLAB_PYTHON}" -m pip install --no-deps --editable /opt/robot_lab_rsl_rl

# Install the project task packages into the Isaac Lab Python environment.
# The base image supplies Isaac Lab and simulator dependencies.
RUN "${ISAACLAB_PYTHON}" -m pip install --no-deps \
      --editable "${ROBOT_LAB_ROOT}/source/robot_learning_lab_datasets" \
      --editable "${ROBOT_LAB_ROOT}/source/robot_learning_lab_zoo[isaaclab]" \
      --editable "${ROBOT_LAB_ROOT}/source/rll_rl" \
      --editable "${ROBOT_LAB_ROOT}/source/robot_learning_lab_tasks[isaaclab]"

# Run as the host UID/GID supplied by Compose. `/tmp` is world writable and
# keeps Kit/Python caches writable without assuming a matching passwd entry.
ENTRYPOINT []
