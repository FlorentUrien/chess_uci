use shared_memory::*;
use std::ptr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shm = ShmemConf::new().os_id("carre_shm").open()?;
    let ptr = shm.as_ptr() as *mut i32;

    println!("C'est parti, j'attends le signal de Python...");

    unsafe {
        loop {
            // On attend que le flag passe à 1
            if ptr::read_volatile(ptr) == 1 {
                for i in 1..6 {
                    let x = ptr::read_volatile(ptr.add(i));
                    // On écrit le carré à l'index i + 5 (zone de sortie)
                    ptr::write_volatile(ptr.add(i + 5), x * x);
                }

                // On met le flag à 2 pour dire à Python que c'est prêt
                ptr::write_volatile(ptr, 2);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}