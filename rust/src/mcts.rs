pub mod node;
pub mod tree;

// use super::conv::policy::get_legal_moves_with_probs;

use crate::conv::board_to_tensor::board_to_tensor;
use crate::conv::from_ia::ia_to_real;
use crate::conv::legaux::coups_legaux;

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
        let max_batch = 256;
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
            println!("Rust : prédictions reçues");

            let value = self.shared_interface.get_value(0);
            self.shared_interface
                .fill_sorted_moves(0, &legaux, &mut self.move_buffer);
            self.shared_interface.reset_flag_prediction();

            println!("Value : {value}");
            for (i, (m_idx, prob)) in self.move_buffer.iter().take(5).enumerate() {
                println!(
                    "{} : {} {}",
                    i + 1,
                    ia_to_real(*m_idx, &board).unwrap(),
                    prob
                );
            }

            // J'ai les mouvements légaux classés par ordre de probabilité décroissante dans move_buffer
            tree.expand_node(0, &self.move_buffer, board);
            tree.profondeur = 1;
        }

        info!(
            "Batch commencé avec {}n / {}p",
            tree.nodes.len(),
            tree.profondeur
        );

        while nb_it < nb_iterations {
            let batch_size = match nb_it {
                0..=299 => 32,
                300..=999 => 64,
                1000..=3999 => 128,
                _ => max_batch, // Le '_' capture tout le reste (le "else")
            };

            // 1. Phase de collecte (Beaucoup plus simple !)
            for _ in 0..batch_size {
                let mut node_index = 0;
                let mut current_prof = 0;
                let mut board_temp = board.clone();
                let mut path: Vec<usize> = Vec::new();

                while tree.nodes[node_index].is_expanded {
                    println!("{} est étendu", node_index);
                    path.push(node_index);
                    tree.nodes[node_index].visit_count += self.virtual_loss;

                    // On enlève le bidouillage de la value_sum, on garde juste la virtual_loss classique
                    // tree.nodes[node_index].value_sum -= self.virtual_loss as f32; // SUPPRIMÉ

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

                println!("{} n'est pas étendu", node_index);

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
                        }

                        for &p_node in &path {
                            tree.nodes[p_node].visit_count -= self.virtual_loss;
                            tree.nodes[p_node].value_sum += self.virtual_loss as f32; // Nettoyage de la pénalité
                        }

                        tree.backpropagate(node_index, v_game, self.virtual_loss);
                        nb_it += 1;
                    } else {
                        // C'est un vrai nouveau nœud à explorer !
                        tree.nodes[node_index].is_pending = true;
                        nodes_exp.push(node_index);
                        board_exp.push(board_temp);
                        println!("{} noeud(s) en attente pour le GPU", nodes_exp.len());
                    }
                }
            }

            if nodes_exp.len() == 0 {
                // Si tout est en attente (pending) et qu'on n'a rien trouvé, on sort
                break;
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

            // 3. Phase de prédiction (GPU)

            let mut size_pred = 0;
            while size_pred == 0 {
                size_pred = self.shared_interface.is_output_ready();
            }
            println!("Rust : {} prédictions reçues", size_pred);

            /*

            // 2. Phase de prédictions (GPU)
            // 2.1 Préparation du batch
            let mut tensors = Vec::with_capacity(board_exp.len());

            for b in &board_exp {
                let fen = Fen::from_position(&b.clone(), EnPassantMode::Always).to_string();
                // 1. Le '?' à la fin extrait le tenseur et gère l'erreur
                let t = self.moteur.fen_to_tensor(&fen)?;
                tensors.push(t);
            }

            // 2.1.1 On prépare des "vues" sur nos petits tenseurs
            let views: Vec<_> = tensors.iter().map(|t| t.view()).collect();

            // 2.1.2 On fusionne tous les (1, 19, 8, 8) en un seul gros tenseur (N, 19, 8, 8)
            // C'est ça que ta Radeon 6650xt va pouvoir traiter d'un coup
            let batch_tensor = ndarray::concatenate(Axis(0), &views)?;
            let taille_batch = batch_tensor.len();

            // 2.2 Injection du batch au GPU
            let debut_gpu = Instant::now();
            let predictions = self.moteur.predict_batch(batch_tensor)?;
            let duree_gpu = debut_gpu.elapsed();
            info!("Batch {}, en {:?}", predictions.len(), duree_gpu);

            // 3. Phase d'expansion
            for i in 0..predictions.len() {
                let v = predictions[i].0;
                let p = &predictions[i].5;
                let coups_legaux = get_legal_moves_with_probs(p, &board_exp[i]);
                // On transforme le Vec<ProbableMove> en Vec<(Move, f32)>
                let coups_pour_arbre: Vec<(Move, f32)> = coups_legaux
                    .iter()
                    .map(|m| (m.mvt.clone(), m.prob))
                    .collect();

                tree.expand_node(nodes_exp[i], &coups_pour_arbre);
                tree.backpropagate(nodes_exp[i], v, self.virtual_loss);
                tree.nodes[nodes_exp[i]].is_pending = false;
                nb_it += 1;
            }
            nodes_exp.clear();
            board_exp.clear();*/
        }
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
