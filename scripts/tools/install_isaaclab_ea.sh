#!/usr/bin/env bash
# Install the Isaac Lab v3.0.0-EA environment (Isaac Sim 6.1.0.0, Python 3.12).
#
# Standard stack for robot_lab:
#   IsaacLab checkout at tag v3.0.0-EA + Isaac Sim 6.1 (pip) + rsl-rl custom 5.x fork
#   (~/humanoid_amp/third_party/rsl_rl, pip name rsl-rl-lib==5.0.1).
#
# Notes:
# - All proxy env vars are stripped: the LAN proxy (100.85.223.50:7890) breaks
#   pypi.org and slows pypi.nvidia.com down.
# - --prerelease=allow: isaacsim deps include rc wheels (opencv-...-noffmpeg).
# - --index-strategy unsafe-best-match: some packages exist on both PyPI and
#   pypi.nvidia.com with different versions.
#
# Usage: ./install_isaaclab_ea.sh [ENV_DIR] [ISAACLAB_DIR]

set -euo pipefail

ENV_DIR="${1:-$HOME/env_isaaclab_ea}"
ISAACLAB_DIR="${2:-$HOME/IsaacLab}"
ROBOT_LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"

# --- 1. venv (Python 3.12, required by EA packages) -------------------------
if [ ! -d "$ENV_DIR" ]; then
    "$UV" venv --python 3.12 "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

NOPROXY="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY"

# --- 2. Isaac Sim 6.1 --------------------------------------------------------
if ! python -c "from importlib.metadata import version; version('isaacsim')" 2>/dev/null; then
    $NOPROXY UV_HTTP_TIMEOUT=1200 "$UV" pip install \
        --prerelease=allow \
        --index-strategy unsafe-best-match \
        --extra-index-url https://pypi.nvidia.com \
        "isaacsim[all,extscache]==6.1.0.0"
fi
python -c "from importlib.metadata import version; print('[OK] isaacsim', version('isaacsim'))"

# --- 3. IsaacLab EA packages (editable; repo must be at v3.0.0-EA) -----------
cd "$ISAACLAB_DIR"
echo "[INFO] IsaacLab at: $(git describe --tags 2>/dev/null || git rev-parse --short HEAD)"
"$UV" pip install --no-deps \
    -e source/isaaclab -e source/isaaclab_physx -e source/isaaclab_rl \
    -e source/isaaclab_tasks -e source/isaaclab_assets
# isaaclab runtime deps (torch etc. pinned by EA pyproject)
$NOPROXY UV_HTTP_TIMEOUT=1200 "$UV" pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url https://pypi.nvidia.com \
    -e source/isaaclab

# --- 4. robot_lab submodules + rsl-rl fork ------------------------------------
cd "$ROBOT_LAB_DIR"
"$UV" pip install --no-deps \
    -e source/robot_learning_lab_tasks \
    -e source/robot_learning_lab_zoo \
    -e source/rll_rl \
    -e "$HOME/humanoid_amp/third_party/rsl_rl"

# --- 5. smoke test -------------------------------------------------------------
echo "[DONE] Env ready: $ENV_DIR"
echo "Verify with:"
echo "  source $ENV_DIR/bin/activate"
echo "  cd $ROBOT_LAB_DIR && python scripts/reinforcement_learning/rsl_rl/play.py \\"
echo "    --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4 --headless"
