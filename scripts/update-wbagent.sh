#!/usr/bin/env bash
set -euo pipefail

upstream_url="https://github.com/wandb/core.git"
upstream_path="services/wb_agent/src/agent_repository/context_content/production/wbagent/skills/wbagent"
skill_dir="skills/wbagent"
commit_file="$skill_dir/.upstream-commit"
target_ref="${1:-master}"

usage() {
  cat <<'EOF'
Usage: scripts/update-wbagent.sh [branch-or-ref]

Sync skills/wbagent from the upstream wbagent skill in wandb/core. The skill is
vendored as plain files in this repo; this script does a shallow sparse clone,
mirrors the upstream directory into skills/wbagent, and records the upstream
commit in skills/wbagent/.upstream-commit. Stage and commit the changes
yourself after reviewing the diff.

Defaults to the upstream master branch.
EOF
}

if [[ "${target_ref}" == "-h" || "${target_ref}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

old_commit="(none)"
if [[ -f "$commit_file" ]]; then
  old_commit=$(cat "$commit_file")
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

git clone --depth 1 --filter=blob:none --sparse --branch "$target_ref" "$upstream_url" "$tmp_dir/core"
git -C "$tmp_dir/core" sparse-checkout set "$upstream_path"
new_commit=$(git -C "$tmp_dir/core" rev-parse HEAD)

src="$tmp_dir/core/$upstream_path"
test -f "$src/SKILL.md"

rm -rf "$skill_dir"
mkdir -p "$skill_dir"
cp -R "$src/." "$skill_dir/"
echo "$new_commit" > "$commit_file"

echo "wbagent synced from ${old_commit:0:12} to ${new_commit:0:12}"
echo
echo "Review the diff with:  git diff -- $skill_dir"
echo "Stage the change with: git add $skill_dir"
