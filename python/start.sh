#!/bin/bash

# Fonction de nettoyage
cleanup() {
    echo "Nettoyage en cours..."
    # On tue Python et tous ses enfants (le tracker compris)
    pkill -P $PYTHON_PID 2>/dev/null
    kill -9 $PYTHON_PID 2>/dev/null
    
    # On force la suppression des fichiers shm si Python a oublié de le faire
    rm -f /dev/shm/shm_tensor /dev/shm/shm_policy /dev/shm/shm_eval /dev/shm/shm_sync /dev/shm/shm_no_model
    
    echo "Tout est propre."
    # N'utilise PAS exit ici si tu comptes encore faire 'source', 
    # mais je te conseille vivement de lancer avec ./start.sh
}

# On dit au script d'appeler cleanup sur les signaux d'arrêt (CTRL+C, kill, etc.)
trap cleanup SIGINT SIGTERM EXIT

# 1. Lancer Python en arrière-plan
cd ~/Rust/chess_uci
source .venv/bin/activate
python python/main.py & 
PYTHON_PID=$!

# 2. Lancer Rust (on attend qu'il finisse)
cd rust
cargo run

# Le script finira par appeler cleanup grâce au trap EXIT