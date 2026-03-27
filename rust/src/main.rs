use std::io::{self, BufRead};
use std::sync::mpsc::{Sender};

mod conv;
mod mcts;
// mod uci;
mod shared_interface;

use crate::mcts::node::Node;
use crate::mcts::tree::MctsTree;
use shakmaty::Chess;

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
    env_logger::init();
    
    println!("Démarrage de la partie Rust");
    let shared_interface = shared_interface::SharedInterface::new(512);
    let mut mcts = mcts::Mcts::new(2.0, 10, shared_interface);
    let new_root = Node::new(None, 1.0);
    let mut new_tree = MctsTree::new(new_root);
    let mut new_board = Chess::new();
    mcts.search_batch(&mut new_tree, &mut new_board, 20000).expect("Ton ku");

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
