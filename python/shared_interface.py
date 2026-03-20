import numpy as np
from time import sleep
from multiprocessing import shared_memory


class SharedInterface:
    """
    Définition de la shared memory entre Python et Rust
    """

    def __init__(self, batch_size=512, create=True):
        """Initialise l'interface de mémoire partagée entre Rust et Python.

        Cette interface gère trois segments principaux :
        1. (Rust -> Python) Tenseurs d'entrée de l'IA (u8) : Format (batch_size, 19, 8, 8).
        2. (Python -> Rust) Policy, les probabilités de coups théoriques possibles (f32) : Format (batch_size, 4672)
        3. (Python -> Rust) Evaluation (f32), le score de la position : Format (batch_size)
        4. (Python <-> Rust) Synch (u8) : Format(2)
        
        Note sur la synchronisation:
            0: Côté Python (0, rien écrit sinon batch_size)
            1: Côté Rust (0, rien écrit sinon batch_size)

        Args:
            batch_size: Taille maximale du batch. Un dépassement causera
                un crash mémoire. (Défaut: 512).
            create: True pour la création (Python), False pour l'attachement (Rust).
        """

        self.batch_size = batch_size

        # Définition des dimensions (doivent matcher Rust !)
        self.shape_tensor = (
            batch_size,
            19,
            8,
            8,
        )  # Les tensors d'entrée du réseau IA (en relatif)
        self.shape_policy = (
            batch_size,
            4672,
        )  # Les probabilités brutes des 4672 coups théoriques possibles (en relatif)
        self.shape_value = (batch_size,)  # L'évaluation de la position (en relatif)
        """
        0: (0) Python n'a pas écrit ses prédictions, sinon taille du batch de prédiction (batch_size)
        1: (0) Rust n'a pas écrit ses tenseurs d'entrée, sinon taille du batch de tenseurs (batch_size)
        """
        self.shape_sync = (2,)

        # Initialisation des segments
        # Note : On utilise 'create=True' seulement côté Python au démarrage
        self.shm_tensor = self._get_shm(
            "shm_tensor",
            np.prod(self.shape_tensor) * np.dtype(np.uint8).itemsize,
            create,
        )
        self.shm_policy = self._get_shm(
            "shm_policy",
            np.prod(self.shape_policy) * np.dtype(np.float32).itemsize,
            create,
        )
        self.shm_eval = self._get_shm(
            "shm_eval",
            np.prod(self.shape_value) * np.dtype(np.float32).itemsize,
            create,
        )
        self.shm_sync = self._get_shm(
            "shm_sync", 2 * np.dtype(np.uint8).itemsize, create
        )

        # Création des vues Numpy (Directement utilisables par ton IA)
        self.tensor = np.ndarray(
            self.shape_tensor, dtype=np.uint8, buffer=self.shm_tensor.buf
        )
        self.policy = np.ndarray(
            self.shape_policy, dtype=np.float32, buffer=self.shm_policy.buf
        )
        self.value = np.ndarray(
            self.shape_value, dtype=np.float32, buffer=self.shm_eval.buf
        )
        self.sync = np.ndarray(
            self.shape_sync, dtype=np.uint8, buffer=self.shm_sync.buf
        )

    def _get_shm(self, name, size, create):
        try:
            return shared_memory.SharedMemory(name=name, create=create, size=int(size))
        except FileExistsError:
            return shared_memory.SharedMemory(name=name)

    def signal_output_ready(self, batch_size: np.ushort):
        """
        Prévient Rust que le GPU a fini l'inférence

        Args:
            batch_size (np.ushort): Taille du batch de prédiction
        """
        self.sync[1] = batch_size

    def wait_for_input(self):
        """Attend que Rust ait rempli le batch de tenseurs"""
        while self.sync[0] == 0:
            pass
        self.sync[0] = 0  # On consomme le signal

    def close(self):
        """Nettoyage propre des segments"""
        for shm in [self.shm_tensor, self.shm_policy, self.shm_eval, self.shm_sync]:
            shm.close()
            if shm.name:  # Optionnel : détruire le segment sur le disque
                try:
                    shm.unlink()
                except Exception:
                    pass
