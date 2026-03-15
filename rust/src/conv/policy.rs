use shakmaty::{Chess, Move, Position};
use super::from_real::real_to_ia;

/// Structure pour stocker un coup légal et sa probabilité associée
pub struct ProbableMove {
    pub mvt: Move,
    pub prob: f32,
}

/// Récupère TOUS les coups légaux avec leurs probabilités respectives classés par probabilité décroissante.
///
/// # Arguments
/// * `proba_move` - Le tableau des probabilité des 4172 coups théoriques
/// * `board` - L'échiquier
///
/// # Retour
/// Retourne le tableau des coups légaux avec leur probabilités normalisées et classé par ordre décroissant.
pub fn get_legal_moves_with_probs(proba_move: &[f32], board: &Chess) -> Vec<ProbableMove> {
    let is_white = board.turn().is_white();
    
    // 1. On récupère TOUS les coups légaux d'abord
    let legals = board.legal_moves();
    
    // 2. On alloue EXACTEMENT la taille nécessaire
    let mut legal_probs = Vec::with_capacity(legals.len());
    
    let mut sum_prob = 0.0;

    for legal_move in legals {
        // Conversion Move -> UciMove pour real_to_ia
        let uci_mvt = legal_move.to_uci(shakmaty::CastlingMode::Standard);
        
        // On récupère l'index IA
        if let Ok(Some(idx)) = real_to_ia(&uci_mvt, is_white) {
            if let Some(&prob) = proba_move.get(idx) {
                sum_prob += prob;
                legal_probs.push(ProbableMove {
                    mvt: legal_move,
                    prob,
                });
            }
        }
    }

    // Normalisation : on divise chaque proba par la somme totale
    // On vérifie sum_prob > 0 pour éviter la division par zéro (cas d'un modèle non entraîné)
    if sum_prob > 0.0 {
        for item in &mut legal_probs {
            item.prob /= sum_prob;
        }
    } else {
        // Fallback : si le modèle est perdu, on distribue équitablement (uniforme)
        let uniform = 1.0 / legal_probs.len() as f32;
        for item in &mut legal_probs {
            item.prob = uniform;
        }
    }
    
    // Tri pour faciliter le travail du MCTS (Exploitation d'abord)
    legal_probs.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));
    
    legal_probs
}