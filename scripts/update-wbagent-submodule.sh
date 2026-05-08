#!/usr/bin/env bash
set -euo pipefail

submodule_path="vendor/wandb-core"
sparse_path="services/wb_agent/src/agent_repository/context_content/production/wbagent/skills/wbagent"
target_ref="${1:-master}"
fetch_depth="${WBAGENT_FETCH_DEPTH:-50}"

usage() {
  cat <<'EOF'
Usage: scripts/update-wbagent-submodule.sh [branch-or-ref]

Fetch the latest upstream wbagent from wandb/core into the sparse submodule.
Defaults to the upstream master branch. Set WBAGENT_FETCH_DEPTH to change the
small commit-history window fetched for review output.
EOF
}

if [[ "${target_ref}" == "-h" || "${target_ref}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

scripts/init-wbagent-submodule.sh >/dev/null

old_commit=$(git -C "$submodule_path" rev-parse HEAD)

git -C "$submodule_path" sparse-checkout init --cone
git -C "$submodule_path" sparse-checkout set "$sparse_path"
git -C "$submodule_path" fetch --filter=blob:none --depth "$fetch_depth" origin "$target_ref"
git -C "$submodule_path" checkout --detach FETCH_HEAD

new_commit=$(git -C "$submodule_path" rev-parse HEAD)

test -f skills/wbagent/SKILL.md
echo "wbagent updated from ${old_commit:0:12} to ${new_commit:0:12}"

if [[ "$old_commit" != "$new_commit" ]]; then
  echo
  echo "Changed upstream wbagent files:"
  git -C "$submodule_path" diff --name-status "$old_commit..$new_commit" -- "$sparse_path" || true
  echo
  echo "Review the full upstream skill diff with:"
  echo "  git -C $submodule_path diff $old_commit..$new_commit -- $sparse_path"
fi

echo "Pin this update with: git add $submodule_path"
