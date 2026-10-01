#!/usr/bin/env bash
# Backwards-compatible path to the portable Docker training launcher.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec "$ROOT/scripts/docker_train.sh" "$@"
