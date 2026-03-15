use shakmaty::{uci::UciMove, Chess, Color, Position, Role, Square};

use anyhow::Ok;

/// On part de l'index prédit, et on le traduit comme un coup dans le monde réel
///
/// # Arguments
/// * `index` - L'indice prédit par le ResNet-10 (ex: 124) [cite: 2026-02-07, 2026-02-14].
/// * `board` - L'échiquier.
///
/// # Retour
/// Retourne le coup au formate UciMove.
pub fn ia_to_real(index: usize, board: &Chess) -> anyhow::Result<UciMove> {
    // 1. On décode l'index (vue IA / perspective trait)
    let move_ia = index_to_move(index, Some(board))
        .ok_or_else(|| anyhow::anyhow!("Index de mouvement invalide : {}", index))?;

    // 2. Si c'est aux Blancs, le coup IA est déjà dans le bon sens
    if board.turn().is_white() {
        return Ok(move_ia);
    }

    // 3. Si c'est aux Noirs, on applique le miroir (^ 56)
    // En shakmaty 0.30, UciMove::Normal contient from, to, et promotion
    match move_ia {
        UciMove::Normal {
            from,
            to,
            promotion,
        } => {
            let flipped_from = from.to_u32() ^ 56;
            let flipped_to = to.to_u32() ^ 56;

            // On reconstruit le mouvement "réel"
            let real_move = UciMove::Normal {
                from: Square::new(flipped_from),
                to: Square::new(flipped_to),
                promotion,
            };
            Ok(real_move)
        }
        _ => Ok(move_ia), // Pour les autres types de coups si nécessaire
    }
}

// On sort les constantes de la fonction
const KNIGHT_MOVES: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const DIRECTIONS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

/// On part le l'index du monde réel pour le traduire en mouvement dans le monde réel
///
/// # Arguments
/// * `index` - L'index dans la base [0-4671]
///
/// # Retour
/// Retourne le coup sous la forme row_depart, col_depart, row_arrivee, col_arrivee.
/// Convertit un index (0-4671) en objet Uci (l'équivalent de chess.Move).
#[inline] // Indique au compilateur d'injecter le code directement là où il est appelé
fn index_to_move(index: usize, board: Option<&Chess>) -> Option<UciMove> {
    if index > 4671 {
        return None;
    }

    let from_sq_idx = (index & 63) as u8; // Utilisation du bitwise & 63
    let plane = index >> 6; // Utilisation du bitwise >> 6 (div 64)

    let from_file = (from_sq_idx & 7) as i8;
    let from_rank = (from_sq_idx >> 3) as i8;

    let (to_file, to_rank, mut promotion);

    // 1. Priorité aux mouvements majoritaires (Reine 0-55)
    if plane < 56 {
        let (df, dr) = DIRECTIONS[plane / 7];
        let distance = ((plane % 7) + 1) as i8;
        to_file = from_file + (df * distance);
        to_rank = from_rank + (dr * distance);
        promotion = None;

        if let Some(b) = board {
            let sq_reel = if b.turn() == Color::White {
                from_sq_idx
            } else {
                from_sq_idx ^ 56
            };
            if let Some(p) = b.board().piece_at(Square::new(sq_reel as u32)) {
                if p.role == Role::Pawn && to_rank == 7 {
                    promotion = Some(Role::Queen);
                }
            }
        }
    }
    // 2. Cavalier (56-63)
    else if plane < 64 {
        let (df, dr) = KNIGHT_MOVES[plane - 56];
        to_file = from_file + df;
        to_rank = from_rank + dr;
        promotion = None;
    }
    // 3. Sous-promotions (64-72)
    else {
        let residue = plane - 64;
        to_file = from_file + ((residue / 3) as i8 - 1);
        to_rank = if from_rank == 6 {
            7
        } else {
            return None;
        };
        promotion = match residue % 3 {
            0 => Some(Role::Knight),
            1 => Some(Role::Bishop),
            _ => Some(Role::Rook),
        };
    }

    if !(0..=7).contains(&to_file) || !(0..=7).contains(&to_rank) {
        return None;
    }

    Some(UciMove::Normal {
        from: Square::new(from_sq_idx as u32),
        to: Square::new(((to_rank << 3) | to_file) as u32), // Bitwise pour reconstruire l'index
        promotion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{fen::Fen, CastlingMode, Chess};

    #[test]
    fn test_reine_e2e4() {
        // Index 124 : Case 12 (e2) + Plan 1 (Direction Nord, distance 4 ?)
        // Note : Ajuste l'index selon TON propre encodage

        let res = index_to_move(124, None);
        // On vérifie si la sortie UCI correspond à ce qu'on attend
        assert_eq!(res, None);
    }

    #[test]
    fn test_g6g7() {
        // Index 124 : Case 12 (e2) + Plan 1 (Direction Nord, distance 4 ?)
        // Note : Ajuste l'index selon TON propre encodage
        let res = index_to_move(46, None);

        assert_eq!(res.unwrap().to_string(), "g6g7");
    }

    #[test]
    fn test_promotion_pion() -> anyhow::Result<()> {
        // On crée un plateau où un pion blanc est en a7
        let fen: Fen =
            "r1bqkbnr/ppp2Qpp/2np4/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4".parse()?;
        let board: Chess = fen.into_position(CastlingMode::Standard)?;

        // Index correspondant à a7 -> a8 avec promotion Dame
        // Dans ton code, c'est un mouvement "Reine" vers le rank 7
        let index_a7a8 = 48; // Exemple : case 48 est a7, plane 0 est Nord distance 1

        let res = index_to_move(index_a7a8, Some(&board));
        assert_eq!(res.unwrap().to_string(), "a7a8q");

        Ok(())
    }

    #[test]
    fn test_dictionnaire_indices() -> anyhow::Result<()> {
        // Ton vecteur de tests
        let tests = vec![
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "f7f5",
                77,
            ),
            (
                "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "e4f5",
                476,
            ),
            (
                "rnbqkbnr/ppppp1pp/8/5P2/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
                "g8g6",
                70,
            ),
            (
                "rnbqkb1r/ppppp1pp/5n2/5P2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3",
                "f1f4",
                133,
            ),
            (
                "rnbqkb1r/pppp2pp/5n2/4pP2/2B5/8/PPPP1PPP/RNBQK1NR w KQkq e6 0 4",
                "f5e6",
                3173,
            ),
            (
                "rnbqk2r/pppp2pp/4Pn2/2b5/2B5/2N5/PPPP1PPP/R1BQK1NR b KQkq - 2 5",
                "e8g8",
                964,
            ),
            (
                "rnbq2k1/pppP1rpp/5n2/2b5/2B5/2N5/PPPP1PPP/R1BQK1NR w KQ - 1 7",
                "d7c8q",
                3187,
            ),
            (
                "rnQq2k1/pp3rp1/2p2n2/2b5/2B5/1PN5/P1PP1PRp/R1BQK1N1 b Q - 0 12",
                "h2g1b",
                4215,
            ),
            (
                "r1Qq2k1/3n1rp1/ppp2n2/2b5/2B2BQ1/1PNP4/P1P2PR1/R3K1b1 w Q - 2 16",
                "e1c1",
                2756,
            ),
            (
                "r1Qq2k1/3n1rp1/ppp2n2/2b5/2B2BQ1/1PNP4/P1P2PR1/2KR2b1 b - - 3 16",
                "d8f8",
                963,
            ),
        ];

        for (fen_str, expected, idx) in tests {
            // 1. On parse le texte en structure Fen (renvoie Result -> .expect)
            let fen: Fen = fen_str.parse()?;

            // 2. On convertit la Fen en position Chess (renvoie Result -> .expect)
            let board: Chess = fen.into_position(CastlingMode::Standard)?;

            // 3. Appel de ta fonction ia_to_real
            let res = ia_to_real(idx, &board)?;

            assert_eq!(res.to_string(), expected, "Échec pour l'index {}", idx);
        }
        Ok(())
    }
}
