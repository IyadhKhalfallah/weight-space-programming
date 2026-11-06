#!/bin/sh
# Run a Python script with this repo on PYTHONPATH. Uses ./.venv if present, else python3.
D="$(cd "$(dirname "$0")" && pwd)"
PY="$D/.venv/bin/python"; [ -x "$PY" ] || PY="$(command -v python3)"
# macOS: XGBoost needs an OpenMP runtime; reuse the one bundled with PyTorch instead of installing libomp
TL="$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)"
[ -n "$TL" ] && export DYLD_LIBRARY_PATH="$TL${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
PYTHONPATH="$D${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" "$@"
