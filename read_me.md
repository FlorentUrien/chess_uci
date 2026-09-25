# Préambule

A ce stade le réseau de neurones a été éduqué préalablement en Python. Il sait analyser une position. Il faut maintenant que l'ordinateur puisse jouer une partie.

J'utilise pour choisir entre les différents coups possibles l'algo du **MCTS** (*Monte Carlo Training Search*).

Pour pouvoir être efficace dans ce parcours arborescent la partie MCTS est écrite dans un langage performant mais complexe à maîtriser le **RUST**, la partie réseau de neurones pure restant en **Python**.

Comme 2 langages différents sont utilisés, l'éditeur ne sera pas **Pycharm** mais **VS code**.

Ces deux parties distinctes communiquent via une <mark>shared memory</mark>.

# Le lanceur des parties Python et Rust

C'est le batch <mark>/python/start.sh</mark>.

On génèrera le fichier de debug <mark>/engine_debug.log</mark>.

## Test de fonctionnement

On lance le moteur : 

`((.venv) ) florent@flo-fixe:~/Rust/chess_uci/python$ source start.sh`

On attend qu'il charche le réseau de neurones et qu'il soit prêt.

`Démarrage de la partie Rust
info string Attente de la mémoire partagée (lancez le script Python)...
info string Mémoire partagée connectée !
Lancement partie Python
Numéro du modèle lu 1
info string Python <- Rust (Je connecte mon model n° 1)
info string Chargement du Modèle /home/florent/Rust/chess_uci/python/models/RN_T6_24012.1_20_best_chess_model_ep45_mae_0.1043_pol_0.4933.weights
Chargement de l'accélérateur ONNX : /home/florent/Rust/chess_uci/python/models/RN_T6_24012.1_20_best_chess_model_ep45_mae_0.1043_pol_0.4933.weights.onnx
info string Le chargement du Modèle /home/florent/Rust/chess_uci/python/models/RN_T6_24012.1_20_best_chess_model_ep45_mae_0.1043_pol_0.4933.weights a réussi`

On initialise le mode <mark>uci</mark>.

`uci
<- uci
id name Florent_IA
option name Simulations type spin default 800 min 100 max 100000
option name PUCT_x10 type spin default 30 min 10 max 50
option name Temperature_x10 type spin default 20 min 1 max 100
option name Limite_stochastique type spin default 5 min 1 max 20
option name Poids_material_x10 type spin default 1 min 0 max 10
option name Poids_echecs_x10 type spin default 1 min 0 max 10
option name Numero_model type spin default 1 min 1 max 10
uciok`

On vérifie que le moteur est prêt.

`isready
<- isready
readyok`

On démarre de la position initiale.

`position startpos
<- position startpos
info string Position mise à jour, coup n°1`

Puis on lui demande de calculer son coup.



# Le coordinateur (Cutechess)

Il faut l'installer, ce qui n'est pas passionnant sous Fedora.

`sudo dnf install -y cmake gcc-c++ qt6-qtbase-devel qt6-qtsvg-devel git make && git clone https://github.com/cutechess/cutechess.git && cd cutechess && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)`

Puis on finalise l'installation :

`sudo make install`

Et enfin on peut le lancer via :

`florent@flo-fixe:~/cutechess/build$ cutechess`

Joie, on a enfin un affichage graphique:

![](images/2026-09-25-09-31-18-image.png)



# La partie Python

![](images/2026-09-25-08-47-16-image.png)

## Le réseau de neurones

Il est stocké sous le répertoire <mark>/models</mark>.

Dans cet exemple on a un réseau Resnet, 6 têtes, appris sur 24 millions de position.

## Les 4 fichiers Python

### main.py

C'est le lanceur de la partie Python (*création de la shared memory et chargement du réseau de neurones*). Ensuite elle organise la communication entre Rust et Python.



## 
