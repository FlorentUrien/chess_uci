import os
import time
import threading
from shared_interface import SharedInterface
from moteur_ia import charger_modele
from multiprocessing import shared_memory
# from predictor import ONNXPredictor


def auto_suicide_if_orphan():
    """Vérifie si le parent est mort et nettoie tout le bordel."""
    # On récupère l'ID du parent au démarrage
    parent_pid = os.getppid()
    
    while True:
        # Sous Linux, si le parent meurt, le PPID devient 1 (processus init)
        if os.getppid() != parent_pid or os.getppid() == 1:
            print("Parent perdu. Nettoyage de la mémoire partagée...")
            # On nettoie manuellement les segments pour éviter les leaks
            for name in ["shm_tensor", "shm_policy", "shm_eval", "shm_sync", "shm_no_model"]:
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    shm.close()
                    shm.unlink()
                except:
                    pass
            os._exit(0) # Fermeture brutale et propre
        time.sleep(1) # On vérifie chaque seconde


def main():
    threading.Thread(target=auto_suicide_if_orphan, daemon=True).start()
    print("Lancement partie Python")
    # 1. On initialise l'interface (c'est Python qui 'create=True' les segments)
    # Assure-toi que le batch_size correspond à celui de ton code Rust
    interface = SharedInterface(batch_size=512, create=True)

    # 2. Il va falloir choisir le bon modèle ONNX à ouvrir
    no_model = interface.wait_for_no_model()

    if no_model == 1:
        path_model = os.path.join(
            ".",
            "models",
            "RN_T6_24012.1_20_best_chess_model_ep45_mae_0.1043_pol_0.4933.weights.onnx",
        )
        charger_modele(path_model)

    try:
        while True:
            # 3. On attend que Rust dise "J'ai fini d'écrire les positions"
            # Cette fonction bloque jusqu'à ce que sync[0] != 0
            interface.wait_for_input()

            # 4. Récupération des données depuis la mémoire partagée
            # L'interface te donne déjà une vue Numpy (self.tensor)
            input_tensor = interface.tensor
            print("tensor reçu")
            print(f"{input_tensor}")

            # 5. Inférence (Le moment où le GPU travaille)
            # Predictor retourne un dict avec 'policy', 'value', etc.
            # predictions = predictor.predict(input_tensor)

            # 6. On recopie les résultats dans la mémoire partagée pour Rust
            # On utilise [:] pour modifier le contenu du segment sans casser la vue
            # interface.policy[:] = predictions['policy']
            # interface.value[:] = predictions['value']

            # 7. On signale à Rust que c'est prêt
            # On lui renvoie la taille du batch pour qu'il sache combien lire
            # interface.signal_output_ready(batch_size=512)

    except KeyboardInterrupt:
        print("\nArrêt de l'IA...")
    finally:
        interface.close()


if __name__ == "__main__":
    main()
