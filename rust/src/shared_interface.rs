use shared_memory::*;

pub struct SharedInterface {
    shm_tensor: Shmem,   // (B, 19, 8, 8) u8
    shm_legaux: Shmem,   // (B, 584)
    shm_policy: Shmem,   // (B, 4672) f32
    shm_value: Shmem,    // (B) f32
    shm_sync: Shmem,     // (2) u16
    shm_no_model: Shmem, // u8
    batch_size: usize,
}

impl SharedInterface {
    pub fn new(batch_size: usize) -> Self {
        println!("info string Attente de la mémoire partagée (lancez le script Python)...");

        // Fonction locale pour tenter d'ouvrir un segment avec répétition
        fn open_shm(id: &str) -> Shmem {
            loop {
                match ShmemConf::new().os_id(id).open() {
                    Ok(m) => return m,
                    Err(_) => {
                        // On attend 500ms avant de réessayer pour ne pas saturer le CPU
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                }
            }
        }

        // On ouvre chaque segment patiemment
        let shm_tensor = open_shm("shm_tensor");
        let shm_legaux = open_shm("shm_legaux");
        let shm_policy = open_shm("shm_policy");
        let shm_value = open_shm("shm_eval");
        let shm_sync = open_shm("shm_sync");
        let shm_no_model = open_shm("shm_no_model");

        println!("info string Mémoire partagée connectée !");

        SharedInterface {
            shm_tensor,
            shm_legaux,
            shm_policy,
            shm_value,
            shm_sync,
            shm_no_model,
            batch_size,
        }
    }

    /// Une méthode propre pour lire la "Value" d'un index précis du batch
    pub fn get_value(&self, batch_idx: usize) -> f32 {
        unsafe {
            let ptr = self.shm_value.as_ptr() as *const f32;
            std::ptr::read_volatile(ptr.add(batch_idx))
        }
    }

    /// Récupère la "Policy" (les probabilités de coups) pour un index du batch
    pub fn get_policy(&self, batch_idx: usize) -> Vec<f32> {
        // 1. On prépare un vecteur pour accueillir les 4672 scores
        let mut policy = vec![0.0f32; 4672];

        unsafe {
            // 2. On récupère le pointeur vers le début du segment de policy
            let ptr = self.shm_policy.as_ptr() as *const f32;

            // 3. On calcule le décalage (chaque entrée du batch fait 4672 flottants)
            let offset_ptr = ptr.add(batch_idx * 4672);

            // 4. On copie proprement la mémoire du segment vers notre vecteur Rust
            std::ptr::copy_nonoverlapping(offset_ptr, policy.as_mut_ptr(), 4672);
        }
        policy
    }

    /// Pour récupéter les mouvements légaux classés par probabilité décroissante
    pub fn get_sorted_moves(&self, batch_idx: usize, mask: &[u8; 584]) -> Vec<(usize, f32)> {
        let raw_policy = self.get_policy(batch_idx);
        let mut legal_scored = Vec::with_capacity(64);

        for idx in 0..4672 {
            // On vérifie le bit dans le masque
            if (mask[idx / 8] & (1 << (idx % 8))) != 0 {
                legal_scored.push((idx, raw_policy[idx]));
            }
        }

        // Tri par score décroissant
        legal_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        legal_scored
    }

    /// Vérifier si Python a fini (Flag Sync)
    ///
    /// # Retour
    ///     =0 : pas fini
    ///     >0 : la taille du batch de prédictions
    pub fn is_output_ready(&self) -> u16 {
        unsafe {
            let ptr = self.shm_sync.as_ptr() as *const u16;
            std::ptr::read_volatile(ptr)
        }
    }

    /// Ecris les tenseurs à passer à l'IA
    ///
    /// # Arguments
    /// * `tensor` - Les tensors aplatis
    /// * `legaux` - Les coups légaux aplatis
    /// * `batch_size` - La taille du batch
    ///
    /// # Retour
    pub fn write_tensors(&self, tensor: *const u8, legaux: *const u8, batch_size: u16) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                tensor,
                self.shm_tensor.as_ptr(),
                1216 * batch_size as usize,
            );
            self.write_legaux(legaux, batch_size);
            let sync_ptr = self.shm_sync.as_ptr() as *mut u16;
            std::ptr::write_volatile(sync_ptr.add(1), batch_size);
        }
    }

    /// Ecris les tenseurs à passer à l'IA
    ///
    /// # Arguments
    /// * `legaux` - Les coups légaux aplatis
    /// * `batch_size` - La taille du batch
    ///
    /// # Retour
    fn write_legaux(&self, legaux: *const u8, batch_size: u16) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                legaux,
                self.shm_legaux.as_ptr(),
                584 * batch_size as usize,
            );
        }
    }

    /// Ecris le modèle ONNX que l'on va utiliser
    ///
    /// # Arguments
    /// * `no` - Le numéro du modèle que Python va utiliser
    pub fn write_no_model(&self, no: u8) {
        unsafe {
            // 1. On récupère le pointeur vers le début du segment mémoire
            let ptr = self.shm_no_model.as_ptr() as *mut u8;

            // 2. On écrit la valeur à l'adresse pointée
            // write_volatile est préférable pour la mémoire partagée pour éviter
            // que le compilateur n'optimise (supprime) l'écriture.
            std::ptr::write_volatile(ptr, no);
        }
    }
}
