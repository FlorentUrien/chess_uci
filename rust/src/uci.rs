use std::io::{self, Write};

/// Lecture de la ligne de commande uci
///
/// # Arguments
/// * `input` - La ligne de commande extraite de std::io
fn lecture_io(input: &mut String) {
    input.clear();
    io::stdin().read_line(input).expect("Erreur");
}

/// Lit les commandes uci sur le std::io
///
/// # Retour
/// * true = si on continue, false si on arrête
pub fn lit_uci() -> bool {
    let mut ligne = String::new();
    let mut ret: bool = true;

    lecture_io(&mut ligne);

    match ligne.trim() {
        "uci" => {
            println!("id name Florent_IA");
            println!("option name Dataset type spin default 1 min 1 max 3");
            println!("option name Blocks_5 type spin default 1 min 1 max 5");
            println!("option name Simulations type spin default 800 min 100 max 100000");
            println!("option name Batch_size type spin default 8 min 4 max 128");
            println!("option name PUCT_x10 type spin default 30 min 10 max 50");
            println!("option name Temperature_x10 type spin default 20 min 1 max 100");
            println!("option name Limite_stochastique type spin default 5 min 1 max 20");
            println!("option name Poids_material_x10 type spin default 1 min 0 max 10");
            println!("option name Poids_echecs_x10 type spin default 1 min 0 max 10");
            println!("option name ONNX type spin default 1 min 0 max 1");
            println!("uciok");
            io::stdout().flush().unwrap();
        }
        "isready" => {
            println!("readyok");
            io::stdout().flush().unwrap(); // VITAL pour Cutechess
        }
        "quit" => ret = true,
        _ => {}
    }
    ret
}

// Pour test uniquement
pub fn loop_uci() {
    loop {
        if !lit_uci(){
        break;
        }
    }
}
