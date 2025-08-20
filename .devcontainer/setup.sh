#!/usr/bin/env bash

# Stop on error
set -e

# ===============
# Fix permissions
# ===============
sudo chown -R few:few /workspaces

# ====================
# Configure pre-commit
# ====================
uv tool install pre-commit --with pre-commit-uv
uv tool run pre-commit install \
  --hook-type commit-msg \
  --hook-type pre-commit \
  --hook-type pre-push
