use shakmaty::{Role, uci::UciMove};

use anyhow::Ok;

#[inline]
/// On part d'un mouvement dans le réel et on récupère l'index de l'ia
///
/// # Arguments
/// * `m` - Le mouvement à traduire
/// * `is_white` - `true` pour les Blancs, `false` pour les Noirs.
///
/// # Retour
/// Retourne l'index ia du coup
pub fn real_to_ia(m: &UciMove, is_white: bool) -> anyhow::Result<Option<usize>> {
    // 1. Extraction du coup
    let from_sq = m.from().map(|s| s as i8).unwrap_or(0);
    let to_sq=m.to().map(|s| s as i8).unwrap_or(0);

    // 2. Miroir immédiat si c'est aux Noirs (Vu de l'IA, on va toujours vers le haut)
    let (from_val, to_val) = if is_white {
        (from_sq, to_sq)
    } else {
        (from_sq ^ 56, to_sq ^ 56) // XOR 56 magique
    };

    // Bitwise pur pour la vitesse : >> 3 au lieu de / 8, et & 7 au lieu de % 8
    let from_rank = from_val >> 3;
    let from_file = from_val & 7;
    let to_rank = to_val >> 3;
    let to_file = to_val & 7;

    let delta_rank = to_rank - from_rank;
    let delta_file = to_file - from_file;

    let plane: usize;

    // 3. Est-ce une sous-promotion ? (Cavalier, Fou, Tour)
    if let Some(role) = m.promotion() {
        if role != Role::Queen {
            let promo_idx = match role {
                Role::Knight => 0,
                Role::Bishop => 1,
                Role::Rook => 2,
                _ => return Ok(None),
            };

            let dir_idx = delta_file + 1; // -1 (gauche) -> 0, 0 (centre) -> 1, 1 (droite) -> 2
            plane = 64 + (dir_idx as usize * 3) + promo_idx;

            return Ok(Some((plane << 6) | (from_val as usize))); // (plane * 64) + from_val
        }
    }

    // 4. Mouvements de Cavalier
    let abs_r = delta_rank.abs();
    let abs_f = delta_file.abs();

    if (abs_r == 2 && abs_f == 1) || (abs_r == 1 && abs_f == 2) {
        // Pattern matching instantané en O(1)
        plane = match (delta_file, delta_rank) {
            (1, 2) => 56,
            (2, 1) => 57,
            (2, -1) => 58,
            (1, -2) => 59,
            (-1, -2) => 60,
            (-2, -1) => 61,
            (-2, 1) => 62,
            (-1, 2) => 63,
            _ => return Ok(None),
        };
    }
    // 5. Mouvements "Reine" (Lignes et Diagonales)
    else {
        let dist = abs_r.max(abs_f);
        if dist == 0 {
            return Ok(None);
        } // Sécurité anti-coup nul

        let sign_r = delta_rank.signum();
        let sign_f = delta_file.signum();

        // Correspondance exacte avec ton tableau DIRECTIONS Rust
        let dir_idx = match (sign_f, sign_r) {
            (0, 1) => 0,   // N
            (1, 1) => 1,   // NE
            (1, 0) => 2,   // E
            (1, -1) => 3,  // SE
            (0, -1) => 4,  // S
            (-1, -1) => 5, // SW
            (-1, 0) => 6,  // W
            (-1, 1) => 7,  // NW
            _ => return Ok(None),
        };

        plane = (dir_idx * 7) + (dist as usize - 1);
    }

    // CALCUL FINAL (plane * 64) + from_val avec opérations binaires
    Ok(Some((plane << 6) | (from_val as usize)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::from_ia::ia_to_real;
    use shakmaty::{Chess, fen::Fen, CastlingMode};

    #[test]
    fn fn_ar()->anyhow::Result<()> {
        let tests = vec![
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                false,
                "f7f5"
            ),
            (
                "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                true,
                "e4f5"
            ),
            (
                "rnbqkbnr/ppppp1pp/8/5P2/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
                false,
                "g8g6"
            ),
            (
                "rnbqkb1r/ppppp1pp/5n2/5P2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3",
                true,
                "f1f4"
            ),
            (
                "rnbqkb1r/pppp2pp/5n2/4pP2/2B5/8/PPPP1PPP/RNBQK1NR w KQkq e6 0 4",
                true,
                "f5e6"
            ),
            (
                "rnbqk2r/pppp2pp/4Pn2/2b5/2B5/2N5/PPPP1PPP/R1BQK1NR b KQkq - 2 5",
                false,
                "e8g8"
            ),
            (
                "rnbq2k1/pppP1rpp/5n2/2b5/2B5/2N5/PPPP1PPP/R1BQK1NR w KQ - 1 7",
                true,
                "d7c8q"
            ),
            (
                "rnQq2k1/pp3rp1/2p2n2/2b5/2B5/1PN5/P1PP1PRp/R1BQK1N1 b Q - 0 12",
                false,
                "h2g1b"
            ),
            (
                "r1Qq2k1/3n1rp1/ppp2n2/2b5/2B2BQ1/1PNP4/P1P2PR1/R3K1b1 w Q - 2 16",
                true,
                "e1c1"
            ),
            (
                "r1Qq2k1/3n1rp1/ppp2n2/2b5/2B2BQ1/1PNP4/P1P2PR1/2KR2b1 b - - 3 16",
                false,
                "d8f8"
            ),
        ];
        for (fen_str,is_white, mouve) in tests {
            let mvt: UciMove = mouve.parse()?;
            let index=real_to_ia(&mvt, is_white)?;
            let idx=index.map(|i|i as usize).unwrap_or(0);

            let fen: Fen = fen_str.parse()?;
            let board: Chess = fen.into_position(CastlingMode::Standard)?;
            let mvt1=ia_to_real(idx, &board)?;


            assert_eq!(mvt1.to_string(), mouve, "Échec pour le mouvement {}", mouve);
        }
        Ok(())
    }
}
