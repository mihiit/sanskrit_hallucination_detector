#!/usr/bin/env bash
#
# git_push.sh
#
# Initializes a local git repository for sanskrit_hallucination_detector
# and pushes it to a remote you specify. Safe to re-run: it only runs
# `git init` if a .git directory doesn't already exist, and only adds
# the remote if it isn't already configured.
#
# Usage:
#   ./git_push.sh <remote-url> [branch-name]
#
# Example:
#   ./git_push.sh git@github.com:yourname/sanskrit_hallucination_detector.git main

set -euo pipefail

REMOTE_URL="${1:-}"
BRANCH="${2:-main}"

if [[ -z "$REMOTE_URL" ]]; then
  echo "Usage: $0 <remote-url> [branch-name]" >&2
  echo "Example: $0 git@github.com:yourname/sanskrit_hallucination_detector.git main" >&2
  exit 1
fi

if ! command -v git &> /dev/null; then
  echo "Error: git is not installed or not on PATH." >&2
  exit 1
fi

if [[ ! -d ".git" ]]; then
  echo "Initializing new git repository..."
  git init
else
  echo "Existing git repository found — skipping init."
fi

# Write a .gitignore if one doesn't already exist, so venvs/caches/
# local resource overrides aren't accidentally committed. (This repo
# ships one by default — this is just a safety net if it's missing.)
if [[ ! -f ".gitignore" ]]; then
  cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.pytest_cache/
settings_local.py
.env
data/resources/*.json
EOF
  echo "Wrote default .gitignore"
fi

git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "Initial commit: sanskrit_hallucination_detector scaffold"
fi

git branch -M "$BRANCH"

if git remote get-url origin &> /dev/null; then
  echo "Remote 'origin' already configured — updating URL."
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

echo "Pushing to $REMOTE_URL (branch: $BRANCH)..."
git push -u origin "$BRANCH"

echo "Done."
