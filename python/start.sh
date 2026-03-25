#!/bin/bash
# On se place au bon endroit
cd "$(dirname "$0")/.."

# 1. On lance Python en arrière-plan (il se surveillera tout seul)
source .venv/bin/activate
python python/main.py &

# 2. On lance Rust en PREMIER PLAN
# Comme c'est lui le parent de Python, s'il meurt, Python suit.
# Redirige la sortie standard (1) et les erreurs (2) vers un fichier log
cd rust
cargo run --release >> ../engine_debug.log 2>&1