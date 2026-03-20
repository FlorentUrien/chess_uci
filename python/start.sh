#!/bin/bash

# 1. Lancer Python en arrière-plan
source .venv/bin/activate
python python/main.py & 
PYTHON_PID=$!

# 2. Attendre 2 secondes que la mémoire soit allouée
sleep 2

# 3. Lancer Rust
cargo run

# 4. Quand Rust s'arrête, on tue Python
kill $PYTHON_PID