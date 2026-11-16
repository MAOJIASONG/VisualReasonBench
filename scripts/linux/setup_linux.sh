#!/usr/bin/env bash
# One-time setup on an x86_64 Linux machine (Ubuntu/Debian).
# Run from the project root:  bash scripts/linux/setup_linux.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "== 1. System packages for the headless Unity player (needs sudo) =="
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    xvfb xauth mesa-utils \
    libgl1 libglu1-mesa libegl1 \
    libxcursor1 libxrandr2 libxinerama1 libxi6 libxkbcommon0 \
    libasound2 libgtk-3-0 || true
else
  echo "apt-get not found: install xvfb + OpenGL/X11 libs with your package manager."
fi

echo "== 2. Unity build =="
if [ "$(uname -m)" != "x86_64" ]; then
  echo "ERROR: Luban/LuBanSuoEnv.x86_64 needs an x86_64 machine (found $(uname -m))." >&2
  exit 1
fi
if [ ! -f Luban/LuBanSuoEnv.x86_64 ]; then
  echo "ERROR: Luban/LuBanSuoEnv.x86_64 missing. Copy the Linux Unity build into Luban/." >&2
  exit 1
fi
chmod +x Luban/LuBanSuoEnv.x86_64

echo "== 3. Python package (editable install into the active env) =="
python -m pip install -e .

echo "== 4. API credentials =="
if [ ! -f .env ]; then
  cat > .env <<'EOF'
# Fill in, then keep this file private (it is gitignored).
OPENAI_API_KEY=
OPENAI_BASE_URL=https://openrouter.ai/api/v1
EOF
  echo "Created .env template — fill in OPENAI_API_KEY."
fi

echo "Setup done. Smoke-test the puzzle env with:  bash scripts/linux/run_luban_linux.sh --smoke"
