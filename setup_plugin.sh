#!/usr/bin/env bash
# One-time setup: link the species/field yt plugin into yt's plugin location so that
# `yt.enable_plugins()` (called by the generator) picks up `load_for_osiris`.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC="$REPO_DIR/flash_osiris/yt_plugin.py"
YT_CFG_DIR="$HOME/.config/yt"
PLUGIN_DST="$YT_CFG_DIR/my_plugins.py"

mkdir -p "$YT_CFG_DIR"
if [[ -e "$PLUGIN_DST" && ! -L "$PLUGIN_DST" ]]; then
    echo "Refusing to overwrite existing non-symlink $PLUGIN_DST" >&2
    echo "Move it aside first, then re-run." >&2
    exit 1
fi
ln -sfn "$PLUGIN_SRC" "$PLUGIN_DST"
echo "Linked $PLUGIN_DST -> $PLUGIN_SRC"
