#!/bin/bash
rm /dev/shm/psm_* 2>/dev/null || true

# On se place au bon endroit
cd "$(dirname "$0")/.."

# On définit le fichier de log une seule fois
LOG_FILE="engine_debug.log"
rm "$LOG_FILE" 2>/dev/null || true

# 1. On lance Python en arrière-plan
# Le ">> $LOG_FILE 2>&1" envoie stdout et stderr dans le log
source .venv/bin/activate
python -u python/main.py >> "$LOG_FILE" 2>&1 &

# 1b. Petite pause pour laisser Python charger les modèles
sleep 2

# 2. On lance Rust en PREMIER PLAN
# On utilise ">>" pour ajouter à la suite du fichier créé par Python
cd rust
cargo run --release >> "../$LOG_FILE" 2>&1