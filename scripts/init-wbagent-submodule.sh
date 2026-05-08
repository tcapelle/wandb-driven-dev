#!/usr/bin/env bash
set -euo pipefail

submodule_path="vendor/wandb-core"
submodule_url="https://github.com/wandb/core.git"
sparse_path="services/wb_agent/src/agent_repository/context_content/production/wbagent/skills/wbagent"

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

pinned_commit=$(git rev-parse ":$submodule_path")

if [[ ! -e "$submodule_path/.git" ]]; then
  if [[ -e "$submodule_path" ]] && [[ -n "$(find "$submodule_path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty $submodule_path" >&2
    exit 1
  fi

  rm -rf "$submodule_path"
  git clone --depth 1 --filter=blob:none --sparse --no-checkout "$submodule_url" "$submodule_path"
fi

git -C "$submodule_path" sparse-checkout init --cone
git -C "$submodule_path" sparse-checkout set "$sparse_path"
git -C "$submodule_path" fetch --depth 1 origin "$pinned_commit"
git -C "$submodule_path" checkout --detach "$pinned_commit"
git submodule absorbgitdirs -- "$submodule_path" >/dev/null 2>&1 || true

test -f skills/wbagent/SKILL.md
echo "wbagent is available at skills/wbagent"
