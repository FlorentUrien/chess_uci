#!/bin/bash
# On se place au bon endroit
cd "$(dirname "$0")/.."

# 1. On lance Python en arrière-plan (il se surveillera tout seul)
source .venv/bin/activate
python python/main.py &

# 2. On lance Rust en PREMIER PLAN
# Comme c'est lui le parent de Python, s'il meurt, Python suit.
cd rust
exec cargo run --release