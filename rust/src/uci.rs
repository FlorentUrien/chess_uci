use shakmaty::uci::UciMove;
use shakmaty::{Chess, Position};
use std::io::{self, Write};
use std::str::FromStr;

use crate::mcts::Mcts;
use crate::shared_interface::SharedInterface;

pub struct Uci {
    simulations: u32, // Nombre de simulations
    batch_size: u16,  // Taille maximale du batch
    puct: f32,
    temperature: f32,
    limite_stochastique: f32,
    poids_material: f32,
    poids_echecs: f32,
    no_model: u8,
    reset: bool,
    pos: Chess,
    force_to_play: bool,
    your_turn: bool,
    shared_mem: SharedInterface,
    mcts: Option<Mcts>,
}

impl Uci {
    /// Constructeur
    pub fn new(shared_mem: SharedInterface) -> Self {
        Self {
            simulations: 800,
            batch_size: 8,
            puct: 2.0,
            temperature: 2.0,
            limite_stochastique: 5.0,
            poids_material: 0.1,
            poids_echecs: 0.1,
            no_model: 0,
            reset: true,
            pos: Chess::default(),
            force_to_play: false,
            your_turn: false,
            shared_mem: shared_mem,
            mcts: None,
        }
    }

    /// Pour indiquer que l'on a consommé le reset
    pub fn reset_done(&mut self) {
        self.reset = false;
    }

    /// Pour indiquer que l'on a consommé le force_to_play
    pub fn reset_force_to_play(&mut self) {
        self.force_to_play = false;
    }

    /// Pour indiquer que l'on a consommé le your_turn
    pub fn reset_your_turn(&mut self) {
        self.your_turn = false;
    }

    /// Mise à jour de la position
    ///
    /// # Arguments
    /// * `line` - La ligne de commande extraite de std::io
    fn update_position(&mut self, ligne: &str) {
        // 0. On peut partir d'une fen
        if ligne.contains("fen") {}
        // 1. On réinitialise si c'est startpos
        if ligne.contains("startpos") {
            self.pos = Chess::default();
        }

        // 2. On traite les coups (moves)
        if let Some(moves_idx) = ligne.find("moves ") {
            // On récupère tout ce qui est après "moves "
            let moves_str = &ligne[moves_idx + 6..];

            for m_str in moves_str.split_whitespace() {
                // Traduction de chess.Move.from_uci(move)
                if let Ok(uci_move) = UciMove::from_str(m_str) {
                    // On convertit en coup légal et on joue
                    if let Ok(m) = uci_move.to_move(&self.pos) {
                        self.pos.play_unchecked(m); // board.push(move)
                    }
                }
            }
        }

        // Calcul du numéro de coup (n // 2)
        let num_coup = self.pos.fullmoves().get();
        println!("info string Position mise à jour, coup n°{}", num_coup);
    }

    /// Lit les commandes uci
    ///
    /// # Arguments
    /// * `ligne` - La ligne de commande uci reçue et à traiter
    ///
    /// # Retour
    /// * true = si on continue, false si on arrête
    pub fn lit_uci(&mut self, ligne: &str) -> bool {
        let mut ret: bool = true;

        match ligne.trim() {
            // Poignée de main initiale, présentation des options configurables
            "uci" => {
                println!("id name Florent_IA");
                println!("option name Simulations type spin default 800 min 100 max 100000");
                println!("option name PUCT_x10 type spin default 30 min 10 max 50");
                println!("option name Temperature_x10 type spin default 20 min 1 max 100");
                println!("option name Limite_stochastique type spin default 5 min 1 max 20");
                println!("option name Poids_material_x10 type spin default 1 min 0 max 10");
                println!("option name Poids_echecs_x10 type spin default 1 min 0 max 10");
                println!("option name Numero_model type spin default 1 min 1 max 10");
                println!("uciok");
                io::stdout().flush().unwrap();
            }
            // Pour annoncer que le réseau de neurones est initialisé
            "isready" => {
                self.shared_mem.write_no_model(self.no_model);
                println!("readyok");
                io::stdout().flush().unwrap(); // VITAL pour Cutechess
            }
            "ucinewgame" => {
                self.pos = Chess::default();
                self.reset = true;
                // TODO : Brancher le MTCS
            }
            "stop" => {
                self.force_to_play = true;
                // TODO : Brancher le MTCS
            }
            "quit" => {
                ret = false;
                // TODO : Brancher la coupure
            }
            _ => {
                // Si ce n'est pas une commande simple, on regarde si ça commence par...
                // setoption (pour changer une option)
                if ligne.starts_with("setoption") {
                    // On cherche les index pour découper proprement la chaîne
                    let name_part = ligne
                        .split("name ")
                        .nth(1)
                        .and_then(|s| s.split(" value").next());
                    let value_part = ligne.split("value ").nth(1);

                    match (name_part, value_part) {
                        (Some(name), Some(value)) => {
                            let name = name.trim();
                            let value = value.trim();

                            match name {
                                "Simulations" => {
                                    if let Ok(v) = value.parse::<u32>() {
                                        self.simulations = v;
                                    }
                                }
                                "Batch_size" => {
                                    if let Ok(v) = value.parse::<u16>() {
                                        self.batch_size = v;
                                    }
                                }
                                "PUCT_x10" => {
                                    if let Ok(v) = value.parse::<f32>() {
                                        self.puct = v / 10.0;
                                    }
                                }
                                "Temperature_x10" => {
                                    if let Ok(v) = value.parse::<f32>() {
                                        self.temperature = v / 10.0;
                                    }
                                }
                                "Limite_stochastique" => {
                                    if let Ok(v) = value.parse::<f32>() {
                                        self.limite_stochastique = v;
                                    }
                                }
                                "Poids_material_x10" => {
                                    if let Ok(v) = value.parse::<f32>() {
                                        self.poids_material = v / 10.0;
                                    }
                                }
                                "Poids_echecs_x10" => {
                                    if let Ok(v) = value.parse::<f32>() {
                                        self.poids_echecs = v / 10.0;
                                    }
                                }
                                "Numero_model" => {
                                    if let Ok(v) = value.parse::<u8>() {
                                        self.no_model = v;
                                    }
                                }
                                _ => println!("info string Option inconnue : {}", name),
                            }
                            println!("info string Option {} mise à jour à {}", name, value);
                        }
                        _ => println!("info string Erreur lors du parsing de l'option"),
                    }
                    // N'oublie pas le flush pour que Cutechess reçoive l'info
                    io::stdout().flush().unwrap();
                }
                // Pour entrer une nouvelle position
                if ligne.starts_with("position") {
                    self.update_position(&ligne);
                }
                // C'est à son tour de jouer
                if ligne.starts_with("go") {
                    // TODO : Traiter nodes xxxx
                    self.your_turn = true;
                }
            }
        } // Fermeture de la session     
        ret
    }
}
