pub mod node;
pub mod tree;

// use super::conv::policy::get_legal_moves_with_probs;

use crate::conv::board_to_tensor::board_to_tensor;
use crate::conv::from_ia::ia_to_real;
use crate::conv::legaux::coups_legaux;

use self::tree::MctsTree;
use crate::shared_interface::SharedInterface;
use nix::libc::sleep;
use shakmaty::Chess;

pub struct Mcts {
    cpuct: f32,
    virtual_loss: u32,
    shared_interface: SharedInterface,
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
        _nb_iterations: u32,
    ) -> anyhow::Result<()> {
        let max_batch = 256;
        let _nb_it: u32 = 0;
        // Pour stocker les noeuds en cours d'expension
        let _nodes_exp: Vec<usize> = Vec::with_capacity(max_batch as usize);
        // Pour stocker les boards des noeuds en cours d'expension
        let _board_exp: Vec<Chess> = Vec::with_capacity(max_batch as usize);

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
            let policy = self.shared_interface.get_sorted_moves(0, &legaux);

            println!("Value : {value}");
            for i in 0..5 {
                println!(
                    "{} : {} {}",
                    i + 1,
                    ia_to_real(policy[i].0, &board).unwrap(),
                    policy[i].1
                );
            }
        }
        /*
                let coups_legaux = get_legal_moves_with_probs(&pred.5, board);
                let coups_pour_arbre: Vec<(Move, f32)> = coups_legaux
                    .iter()
                    .map(|m| (m.mvt.clone(), m.prob)) // On extrait juste ce qu'il faut
                    .collect();
                tree.expand_node(0, &coups_pour_arbre);
                nb_it = 1;
                tree.profondeur = 1;
            }

            info!(
                "Batch commencé avec {}n / {}p",
                tree.nodes.len(),
                tree.profondeur
            );

            while nb_it < nb_iterations {
                let batch_size = match nb_it {
                    0..=99 => 16,
                    100..=299 => 32,
                    300..=999 => 64,
                    1000..=3999 => 128,
                    _ => max_batch, // Le '_' capture tout le reste (le "else")
                };

                // 1. Phase de collecte
                for _ in 0..batch_size {
                    let mut node_index = 0;
                    let mut current_prof = 0;
                    let mut board_temp = board.clone();

                    while tree.nodes[node_index].is_expanded {
                        tree.nodes[node_index].visit_count += self.virtual_loss;
                        let (mv, idx_node) = tree.nodes[node_index]
                            .select_child(&tree.nodes, self.cpuct)
                            .expect("Problème select_child");
                        board_temp = board_temp.play(mv)?;
                        current_prof += 1;
                        node_index = idx_node;
                    }

                    if current_prof > tree.profondeur {
                        tree.profondeur = current_prof;
                        info!("{}n / {}p", tree.nodes.len(), tree.profondeur);
                    }

                    if !tree.nodes[node_index].is_pending {
                        if board_temp.is_game_over() {
                            let valeur_mat: f32 = 2.0;
                            let gamma: f32 = 0.01;
                            let mut v_game: f32 = 0.0;

                            if board_temp.is_checkmate() {
                                v_game = -(valeur_mat - (current_prof as f32 * gamma));
                            }
                            tree.backpropagate(node_index, v_game, self.virtual_loss);
                            nb_it += 1;
                        } else {
                            tree.nodes[node_index].is_pending = true;
                            nodes_exp.push(node_index);
                            board_exp.push(board_temp.clone());
                        }
                    }
                }
                if nodes_exp.len() == 0 {
                    break;
                }

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
                board_exp.clear();
            }
            Ok(())
        }*/
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
