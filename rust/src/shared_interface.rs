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

    /// Une fois les prédictions lues il faut reset le flag
    pub fn reset_flag_prediction(&self) {
        unsafe {
            let sync_ptr = self.shm_sync.as_ptr() as *mut u16;
            std::ptr::write_volatile(sync_ptr, 0);
        }
    }

    /// Une méthode propre pour lire la "Value" d'un index précis du batch
    pub fn get_value(&self, batch_idx: usize) -> f32 {
        unsafe {
            let ptr = self.shm_value.as_ptr() as *const f32;
            std::ptr::read_volatile(ptr.add(batch_idx))
        }
    }

    // On passe une référence mutable au vecteur pour le recycler
    pub fn fill_sorted_moves(
        &self,
        batch_idx: usize,
        mask: &[u8; 584],
        output: &mut Vec<(usize, f32)>,
    ) {
        // On crée une "vue" (slice) sur la mémoire partagée SANS COPIE
        let raw_policy: &[f32] = unsafe {
            let ptr = self.shm_policy.as_ptr() as *const f32;
            let offset_ptr = ptr.add(batch_idx * 4672);
            std::slice::from_raw_parts(offset_ptr, 4672) // Accès direct à la SHM
        };

        // On vide le vecteur sans désallouer sa mémoire
        output.clear();

        for (byte_idx, &byte) in mask.iter().enumerate() {
            if byte == 0 {
                continue;
            }
            for bit_idx in 0..8 {
                if (byte & (1 << bit_idx)) != 0 {
                    let idx = (byte_idx << 3) | bit_idx;
                    unsafe {
                        output.push((idx, *raw_policy.get_unchecked(idx)));
                    }
                }
            }
        }

        output.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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

    /// Vérifier si Python a réussi à charger le réseau
    ///
    /// # Retour
    ///     =0 : pas fini
    ///     =1 : ok
    ///     =2 : pb
    pub fn nn_state(&self) -> u16 {
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
