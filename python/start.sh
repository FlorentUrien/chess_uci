#!/bin/bash
rm /dev/shm/psm_* 2>/dev/null || true

# Récupère le dossier parent du dossier où se trouve le script (donc la racine chess_uci)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# On définit le fichier de log une seule fois
LOG_FILE="engine_debug.log"
rm "$LOG_FILE" 2>/dev/null || true

# 1. On lance Python en arrière-plan
source .venv/bin/activate
python -u python/main.py 2>> "$LOG_FILE" &

# 2. On lance Rust en PREMIER PLAN
# On utilise ">>" pour ajouter à la suite du fichier créé par Python
cd rust
RUST_LOG=info cargo run --release 2>> "../$LOG_FILE"