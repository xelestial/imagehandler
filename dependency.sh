#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf '\033[1;33m[dependency]\033[0m dependency.sh forwards to setup.sh. Use ./setup.sh directly.\n'
"$ROOT_DIR/setup.sh" "$@"
