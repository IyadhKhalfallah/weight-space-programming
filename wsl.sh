#!/bin/sh
# Run the wsl CLI from anywhere
D="$(cd "$(dirname "$0")" && pwd)"
exec "$D/py.sh" -m wsl.cli "$@"
