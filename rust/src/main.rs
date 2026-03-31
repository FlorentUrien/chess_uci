use std::io::{self, BufRead};
use std::sync::mpsc::Sender;

mod conv;
mod mcts;
// mod uci;
mod shared_interface;

use crate::mcts::node::Node;
use crate::mcts::tree::MctsTree;
use shakmaty::{CastlingMode, Chess, fen::Fen};

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
    /*let fen: Fen = "2k4r/ppp1q1b1/B5p1/4np2/8/6r1/PPQ2PPP/R4RK1 b - - 0 18"
    .parse()
    .expect("tonku");*/
    /*let fen: Fen = "2kr2nr/1pp5/p2p1p1b/1n1P4/4P1q1/1QP2NBb/PP1N1P1K/R5R1 b - - 0 18"
    .parse()
    .expect("tonku");*/
    let fen: Fen = "1rbqkbnr/pppppppp/8/8/1nB1P3/5Q2/PPPP1PPP/RNB1K1NR w KQk - 0 3"
        .parse()
        .expect("tonku");
    /*let fen: Fen = "r1bqkbnr/p1pp1ppp/1p6/4p1NQ/1n2P3/8/PPPP1PPP/RNB1KB1R w KQkq - 2 5"
        .parse()
        .expect("tonku");
    let fen: Fen = "2k4r/ppp1q1b1/B5p1/4np2/8/6r1/PPQ2PPP/R4RK1 b - - 0 18"
        .parse()
        .expect("tonku"); // -M5
    let fen: Fen = "2kr2nr/1pp5/p2p1p2/1n1P4/4Pbq1/1QP2NBb/PP1N1P1K/R6R b - - 2 19"
        .parse()
        .expect("tonku"); // -M4 facile à trouver Bh3-f1+*/
    let mut new_board: Chess = fen.into_position(CastlingMode::Standard).expect("tonku");
    // let mut new_board = Chess::new();
    mcts.search_batch(&mut new_tree, &mut new_board, 20000)
        .expect("Ton ku");
    println!(
        "Meilleur coup = {}",
        new_tree.choisir_meilleur_coup().unwrap()
    );
    let meilleurs_coups = new_tree.choisir_x_meilleurs_coups(40);

    if let Some(meilleur) = meilleurs_coups.first() {
        println!("Meilleur coup = {}", meilleur);
    } else {
        println!("Aucun coup trouvé !");
    }

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
