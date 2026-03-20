use shared_memory::*;

struct SharedInterface {
    shm_tensor: Shmem,   // (B, 19, 8, 8) u8
    shm_policy: Shmem,  // (B, 4672) f32
    shm_value: Shmem,   // (B) f32
    shm_sync: Shmem,    // (2) u8
    batch_size: usize,
}

impl SharedInterface {
    fn new(batch_size: usize) -> Self {
        // Ici on ouvre les 4 segments par leurs noms
        let shm_tensor = ShmemConf::new().os_id("shm_tensor").open().unwrap();
        let shm_policy = ShmemConf::new().os_id("shm_policy").open().unwrap();
        let shm_value = ShmemConf::new().os_id("shm_eval").open().unwrap();
        let shm_sync = ShmemConf::new().os_id("shm_sync").open().unwrap();

        SharedInterface {
            shm_tensor,
            shm_policy,
            shm_value,
            shm_sync,
            batch_size,
        }
    }

    // Une méthode propre pour lire la "Value" d'un index précis du batch
    fn get_value(&self, batch_idx: usize) -> f32 {
        unsafe {
            let ptr = self.shm_value.as_ptr() as *const f32;
            std::ptr::read_volatile(ptr.add(batch_idx))
        }
    }

    // Vérifier si Python a fini (Flag Sync)
    fn is_output_ready(&self) -> bool {
        unsafe {
            let ptr = self.shm_sync.as_ptr() as *const u8;
            std::ptr::read_volatile(ptr.add(1)) == 1
        }
    }

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
    fn write_tensors(){
        shm_sync[1]=0;
    }
}