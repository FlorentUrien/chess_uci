use shared_memory::*;

pub struct SharedInterface {
    shm_tensor: Shmem,   // (B, 19, 8, 8) u8
    shm_policy: Shmem,  // (B, 4672) f32
    shm_value: Shmem,   // (B) f32
    shm_sync: Shmem,    // (2) u8
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
        let shm_tensor   = open_shm("shm_tensor");
        let shm_policy   = open_shm("shm_policy");
        let shm_value    = open_shm("shm_eval");
        let shm_sync     = open_shm("shm_sync");
        let shm_no_model = open_shm("shm_no_model");

        println!("info string Mémoire partagée connectée !");

        SharedInterface {
            shm_tensor,
            shm_policy,
            shm_value,
            shm_sync,
            shm_no_model,
            batch_size,
        }
    }

    // Une méthode propre pour lire la "Value" d'un index précis du batch
    /*fn get_value(&self, batch_idx: usize) -> f32 {
        unsafe {
            let ptr = self.shm_value.as_ptr() as *const f32;
            std::ptr::read_volatile(ptr.add(batch_idx))
        }
    }*/

    // Vérifier si Python a fini (Flag Sync)
    /*fn is_output_ready(&self) -> bool {
        unsafe {
            let ptr = self.shm_sync.as_ptr() as *const u8;
            std::ptr::read_volatile(ptr.add(1)) == 1
        }
    }*/

    /// Ecris les tenseurs à passer à l'IA
    ///
    /// # Arguments
    /// * `tree` - L'arbre que l'on va développer
    ///
    /// &mut pour dire qu'on le développe mais qu'on le rend ensuite. Pour ne pas le jeter après usage.
    /// * `board` - L'échiquier
    /// * `nb_iterations` - Le nombre de noeuds à développer
    ///
    /// # Retour
    /*fn write_tensors(&self){
        self.shm_sync[1]=0;
    }*/

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