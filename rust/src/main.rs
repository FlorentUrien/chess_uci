use shared_memory::*;
use std::io::{self, BufRead};
use std::ptr;
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::Duration;

use crate::uci::Uci;

// mod mcts;
// mod conv;
mod uci;

/*fn main() -> Result<(), Box<dyn std::error::Error>> {
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
}*/

/// Gère le thread d'écoute du stdin pour capter les échanges uci avec Cutechess
///
/// # Arguments
/// * `tx` - Le canal de communication avec le père
fn ecoute_stdin(tx: Sender<String>) {
    let stdin = io::stdin();
    // On boucle sur chaque ligne saisie
    for line in stdin.lock().lines() {
        match line {
            Ok(texte) => {
                let _ = tx.send(texte);
            }
            Err(e) => eprintln!("Erreur de lecture : {}", e),
        }
    }
}

fn main() {
    println!("Démarrage du test UCI");

    let mut uci = Uci::new();

    // 0. On crée le canal de communication
    let (tx, rx) = mpsc::channel::<String>();

    // 1. On lance le thread de lecture
    thread::spawn(|| {
        ecoute_stdin(tx);
    });

    // 2. Boucle principale (le programme continue de bosser)
    loop {
        if let Ok(msg) = rx.try_recv() {
            uci.lit_uci(&msg);
        }
        thread::sleep(Duration::from_millis(10));
    }
}
