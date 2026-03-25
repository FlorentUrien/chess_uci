use shared_memory::*;
use std::io::{self, BufRead};
use std::ptr;
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::Duration;

// use crate::uci::Uci;

mod mcts;
mod conv;
// mod uci;
mod shared_interface;


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
    println!("Démarrage de la partie Rust");

/*     let shared_mem = shared_interface::SharedInterface::new(512);
    let mut uci = Uci::new(shared_mem);    

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
    }*/
}
