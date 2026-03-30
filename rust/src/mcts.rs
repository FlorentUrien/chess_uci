pub mod node;
pub mod tree;
use std::time::{Duration, Instant};

// use super::conv::policy::get_legal_moves_with_probs;

use crate::conv::from_ia::ia_to_real;
use crate::conv::legaux::coups_legaux;
use crate::{conv::board_to_tensor::board_to_tensor, shared_interface};

use self::tree::MctsTree;
use crate::shared_interface::SharedInterface;
use log::info;
use nix::libc::sleep;
use shakmaty::{Chess, Position};

pub struct Mcts {
    cpuct: f32,
    virtual_loss: u32,
    shared_interface: SharedInterface,
    move_buffer: Vec<(usize, f32)>,
}

impl Mcts {
    /// Constructeur
    ///
    /// # Arguments
    /// * `cpuct` - Joue sur la largeur de recherche
    /// * `virtual_loss` - 10 peut être une bonne valeur (1 à 100), cela indique la diversité de la recherche pendant la phase d'exploration (en pending)
    ///
    /// Exemple (CPUCT):
    /// * 1.0 on est sûr de ses évaluations, on s'enfonce étroit
    /// * 5.0 au contraire on hésite et on s'étale
    ///
    /// Normalement on choisira entre 2.0 & 3.0.
    ///
    /// # Retour
    /// MonteCarlo Training Search
    pub fn new(cpuct: f32, virtual_loss: u32, shared_interface: SharedInterface) -> Self {
        Self {
            cpuct,
            virtual_loss,
            shared_interface,
            move_buffer: Vec::with_capacity(64),
        }
    }

    /// Gère les nb_iterations recherches en les découpants en batch pour optimiser l'usage GPU
    ///
    /// # Arguments
    /// * `tree` - L'arbre que l'on va développer
    ///
    /// &mut pour dire qu'on le développe mais qu'on le rend ensuite. Pour ne pas le jeter après usage.
    /// * `board` - L'échiquier
    /// * `nb_iterations` - Le nombre de noeuds à développer
    ///
    /// # Retour
    pub fn search_batch(
        &mut self,
        tree: &mut MctsTree,
        board: &mut Chess,
        nb_iterations: u32,
    ) -> anyhow::Result<()> {
        let max_batch = 512;
        let mut nb_it = 1;
        // Pour stocker les noeuds en cours d'expension
        let mut nodes_exp: Vec<usize> = Vec::with_capacity(max_batch as usize);
        // Pour stocker les boards des noeuds en cours d'expension
        let mut board_exp: Vec<Chess> = Vec::with_capacity(max_batch as usize);

        println!("Rust -> Python (connecte ton model n°1)");
        self.shared_interface.write_no_model(1);

        unsafe {
            sleep(5);
        }
        let mut duree_python = 0;
        let mut duree_rust = 0;

        if !(tree.nodes[0].is_expanded) {
            // La racine n'est pas étendue, on ne peut pas avancer avant de l'avoir étendue

            let tensor = board_to_tensor(&board);
            let ptr = tensor.as_ptr() as *const u8;
            let legaux = coups_legaux(&board);
            let ptr_legaux = legaux.as_ptr() as *const u8;
            self.shared_interface.write_tensors(ptr, ptr_legaux, 1);

            // Prédiction GPU
            println!("Rust -> demande de prédiction à Python");
            let mut size_pred: u16 = 0;
            while size_pred == 0 {
                size_pred = self.shared_interface.is_output_ready();
            }
            println!("rootRust : prédictions reçues");

            let value = self.shared_interface.get_value(0);
            tree.nodes[0].value_sum = value;
            self.shared_interface
                .fill_sorted_moves(0, &legaux, &mut self.move_buffer);
            self.shared_interface.reset_flag_prediction();

            // J'ai les mouvements légaux classés par ordre de probabilité décroissante dans move_buffer
            tree.expand_node(0, &self.move_buffer, board);
            tree.profondeur = 1;

            tree.affiche_sonnet();

            //tree.affiche();
        }

        info!(
            "Batch commencé avec {}n / {}p",
            tree.nodes.len(),
            tree.profondeur
        );

        let top = Instant::now();
        let mut duree_python: Duration = Duration::ZERO;

        while nb_it < nb_iterations {
            let batch_size = match nb_it {
                0..=299 => 64,
                300..=999 => 128,
                1000..=3999 => 256,
                _ => max_batch, // Le '_' capture tout le reste (le "else")
            };

            // 1. Phase de collecte
            for _ in 0..batch_size {
                let mut node_index = 0;
                let mut current_prof = 0;
                let mut board_temp = board.clone();
                let mut path: Vec<usize> = Vec::new();

                while tree.nodes[node_index].is_expanded {
                    path.push(node_index);
                    tree.nodes[node_index].visit_count += self.virtual_loss;

                    // Si select_child retourne None, ça veut dire que TOUS les enfants
                    // sont "pending" (en attente GPU). Dans ce cas, on ne peut pas descendre.
                    if let Some((mv, idx_node)) =
                        tree.nodes[node_index].select_child(&tree.nodes, self.cpuct)
                    {
                        board_temp = board_temp.play(mv)?;
                        current_prof += 1;
                        node_index = idx_node;
                    } else {
                        // Tous les enfants de cette branche sont occupés. On casse la boucle pour abandonner.
                        break;
                    }
                }

                // Si on a breaké (ou si c'est déjà pending), c'est qu'on est bloqué.
                if tree.nodes[node_index].is_expanded || tree.nodes[node_index].is_pending {
                    // On nettoie la trace et on passe au ticket suivant
                    for &p_node in &path {
                        tree.nodes[p_node].visit_count -= self.virtual_loss;
                    }
                    println!("Branche saturée on envoit le batch partiel");
                    break;
                }

                if current_prof > tree.profondeur {
                    tree.profondeur = current_prof;
                    info!("{}n / {}p", tree.nodes.len(), tree.profondeur);
                }

                if !tree.nodes[node_index].is_expanded && !tree.nodes[node_index].is_pending {
                    if board_temp.is_game_over() {
                        let valeur_mat: f32 = 2.0;
                        let gamma: f32 = 0.01;
                        let mut v_game: f32 = 0.0;

                        if board_temp.is_checkmate() {
                            v_game = -(valeur_mat - (current_prof as f32 * gamma));
                            println!("Checkmate {}", v_game);
                        }

                        for &p_node in &path {
                            tree.nodes[p_node].visit_count -= self.virtual_loss;
                        }

                        tree.backpropagate(node_index, v_game, self.virtual_loss);
                        nb_it += 1;
                    } else {
                        // C'est un vrai nouveau nœud à explorer !
                        tree.nodes[node_index].is_pending = true;
                        nodes_exp.push(node_index);
                        board_exp.push(board_temp);
                    }
                }
            }

            if nodes_exp.len() == 0 {
                // Si tout est en attente (pending) et qu'on n'a rien trouvé, on sort
                continue;
            }

            // 2. Préparation des données à envoyer via shared memory à Python et donc au GPU

            let current_batch_size = nodes_exp.len();
            let mut batch_tensors: Vec<u8> = Vec::with_capacity(current_batch_size * 1216); // 19x8x8 = 1216
            let mut batch_legal: Vec<u8> = Vec::with_capacity(current_batch_size * 584);

            for b in &board_exp {
                let tensor = board_to_tensor(b);

                // On transforme le tableau 3D en une slice 1D de 1216 octets
                // On utilise unsafe ici car on garantit au compilateur que la structure
                // en mémoire est bien une suite continue de u8.
                let flat_slice: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        tensor.as_ptr() as *const u8,
                        1216, // 19 * 8 * 8
                    )
                };

                batch_tensors.extend_from_slice(flat_slice);
                batch_legal.extend_from_slice(&coups_legaux(b));
            }

            self.shared_interface.write_tensors(
                batch_tensors.as_ptr(),
                batch_legal.as_ptr(),
                current_batch_size as u16,
            );

            let top_python = Instant::now();

            // 3. Phase de prédiction (GPU)

            let mut size_pred = 0;
            while size_pred == 0 {
                size_pred = self.shared_interface.is_output_ready();
            }
            size_pred = size_pred.min(current_batch_size as u16);
            duree_python += Instant::now() - top_python;
            println!("Rust : {} prédictions reçues", size_pred);

            // 4. On étend l'arbre
            for i in 0..size_pred {
                let i_usize = i as usize;
                let start = i_usize * 584;
                let end = start + 584;

                // On récupère la tranche de 584 octets
                let mask_slice = &batch_legal[start..end];

                // On convertit la slice en référence de tableau fixe (le compilateur vérifie la taille)
                let mask_array: &[u8; 584] =
                    mask_slice.try_into().expect("Taille de masque invalide");

                let value = self.shared_interface.get_value(i as usize);
                tree.nodes[nodes_exp[i as usize]].value_sum = value;
                self.shared_interface
                    .fill_sorted_moves(0, mask_array, &mut self.move_buffer);
                tree.expand_node(
                    nodes_exp[i as usize],
                    &self.move_buffer,
                    &board_exp[i as usize],
                );
                tree.backpropagate(nodes_exp[i as usize], value, self.virtual_loss);
                nb_it += 1;
            }
            self.shared_interface.reset_flag_prediction();
            //tree.affiche();
            tree.affiche_sonnet();

            nodes_exp.clear();
            board_exp.clear();
        }
        let duree_totale = Instant::now() - top;
        let duree_rust = duree_totale - duree_python;
        println!(
            "Total : {:?}, Python: {:?}, Rust: {:?}",
            duree_totale, duree_python, duree_rust
        );
        Ok(())
    }
}

/*
#[cfg(test)]
mod tests {
    use shakmaty::{CastlingMode, fen::Fen};

    use super::*;
    use crate::{
        constantes::{MODEL_NAME, WEIGHTS},
        mcts::node::Node,
        moteur,
    };
    use std::path::PathBuf;

    #[test]
    fn test_evaluate_position() -> anyhow::Result<()> {
        // Initialisation (Charge le modèle ONNX)
        let mut model_path = PathBuf::from(WEIGHTS);
        model_path.push(MODEL_NAME);

        let moteur = Moteur::new(model_path, "10M Res10");
        let mut mtcs = Mcts::new(moteur, 2.0, 10);

        let fen_str = "r1bqkbnr/2pppppp/p1n5/1p6/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1";
        let fen: Fen = fen_str.parse()?;
        let board: Chess = fen.into_position(CastlingMode::Standard)?;

        let ret = mtcs.evaluate_position(&board);
        println!("Test M1");
        println!("{}", fen_str);
        let (eval, pol) = ret.expect("L'évaluation a échoué");
        println!("Evaluation : {:.2}", eval);
        for i in 0..5 {
            let (m, prob) = pol[i];
            println!(
                "{}: {} / {:.1}%",
                i + 1,
                m.to_uci(CastlingMode::Standard).to_string(),
                100.0 * prob
            );
        }

        let fen_str = "rnb1k1nr/pppp1ppp/8/1Nb1p3/1P5q/P7/2PPPPPP/R1BQKBNR b KQkq - 0 4";
        let fen: Fen = fen_str.parse()?;
        let board: Chess = fen.into_position(CastlingMode::Standard)?;

        let ret = mtcs.evaluate_position(&board);
        println!("Test -M1");
        println!("{}", fen_str);
        let (eval, pol) = ret.expect("L'évaluation a échoué");
        println!("Evaluation : {:.2}", eval);
        for i in 0..5 {
            let (m, prob) = pol[i];
            println!(
                "{}: {} / {:.1}%",
                i + 1,
                m.to_uci(CastlingMode::Standard).to_string(),
                100.0 * prob
            );
        }

        Ok(())
    }

    #[test]
    fn test_search_batch() -> anyhow::Result<()> {
        std::env::set_var("RUST_LOG", "info");
        let _ = env_logger::builder().is_test(true).try_init();

        let mut model_path = PathBuf::from(WEIGHTS);
        model_path.push(MODEL_NAME);

        let moteur = Moteur::new(model_path, "10M Res10");
        let mut mtcs = Mcts::new(moteur, 2.0, 10);
        let mut tree = MctsTree::new(Node::new(None, 0.0));
        let fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let fen: Fen = fen_str.parse()?;
        let board: Chess = fen.into_position(CastlingMode::Standard)?;

        mtcs.search_batch(&mut tree, &board, 2000);

        Ok(())
    }
}*/
