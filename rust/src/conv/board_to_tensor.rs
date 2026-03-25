use chess::{Board, Color, Piece};
use std::str::FromStr;

/// Conversion du plateau en tenseur 19x8x8.
/// Retourne un tableau 3D de 19 plans, 8 lignes, 8 colonnes.
pub fn board_to_tensor(board: &Board) -> [[[u8; 8]; 8]; 19] {
    // 1. Initialisation du tenseur rempli de zéros
    let mut tensor = [[[0u8; 8]; 8]; 19];

    let turn = board.side_to_move();
    let is_white = turn == Color::White;

    // 2 & 3. Remplir les pièces (0-11) et les masques de présence (17-18)
    // On boucle sur les 64 cases du plateau
    for sq in chess::ALL_SQUARES {
        if let Some(piece) = board.piece_on(sq) {
            let piece_color = board.color_on(sq).unwrap();

            // Calcul du "Flip" vertical si c'est aux Noirs de jouer
            // rank 0 = rangée 1 (A1-H1), rank 7 = rangée 8 (A8-H8)
            let file = sq.get_file().to_index();
            let rank = sq.get_rank().to_index();
            let view_rank = if is_white { rank } else { 7 - rank };

            let piece_idx = match piece {
                Piece::Pawn => 0,
                Piece::Knight => 1,
                Piece::Bishop => 2,
                Piece::Rook => 3,
                Piece::Queen => 4,
                Piece::King => 5,
            };

            if piece_color == turn {
                tensor[piece_idx][view_rank][file] = 1; // Mes pièces
                tensor[17][view_rank][file] = 1; // Présence amie
            } else {
                tensor[piece_idx + 6][view_rank][file] = 1; // Ses pièces
                tensor[18][view_rank][file] = 1; // Présence ennemie
            }
        }
    }

    // 4. Droits au roque (Couches 12-15 : plans pleins)
    let my_ks = board.castle_rights(turn).has_kingside();
    let my_qs = board.castle_rights(turn).has_queenside();
    let his_ks = board.castle_rights(!turn).has_kingside();
    let his_qs = board.castle_rights(!turn).has_queenside();

    // Si on a le droit, on remplit tout le plan de 1
    for r in 0..8 {
        for f in 0..8 {
            if my_ks {
                tensor[12][r][f] = 1;
            }
            if my_qs {
                tensor[13][r][f] = 1;
            }
            if his_ks {
                tensor[14][r][f] = 1;
            }
            if his_qs {
                tensor[15][r][f] = 1;
            }
        }
    }

    // 5. Prise en passant (Couche 16)
    if let Some(ep_sq) = board.en_passant() {
        let file = ep_sq.get_file().to_index();
        let rank = ep_sq.get_rank().to_index();

        // La case EP est la case VIDE derrière le pion.
        // Le pion adverse, lui, est sur le rank 3 (si Blanc joue) ou rank 4 (si Noir joue).
        // On veut marquer la case du PION capturable pour l'IA.
        let pawn_rank = if is_white { rank + 1 } else { rank - 1 };

        // On applique le miroir visuel habituel sur ce rang
        let view_rank = if is_white { pawn_rank } else { 7 - pawn_rank };

        tensor[16][view_rank][file] = 1;
    }

    tensor
}

/// Conversion d'un fen en tenseur 19x8x8.
/// Retourne un tableau 3D de 19 plans, 8 lignes, 8 colonnes.
pub fn fen_to_tensor(fen_str: &str) -> [[[u8; 8]; 8]; 19] {
    // 1. On transforme la chaîne FEN en un objet Board
    // .expect() fera crasher le moteur si la FEN est invalide (rare en UCI)
    let board = Board::from_str(fen_str).expect("FEN invalide");

    // 2. On appelle la fonction de conversion
    board_to_tensor(&board)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fen_to_tensor() {
        /*let fen = "rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR b KQkq - 0 1";
        let tensor = fen_to_tensor(fen);
        // On suppose que 'tensor' est de type [[[u8; 8]; 8]; 19]
        println!("Position de départ: ");

        for i in 0..19 {
            println!("\nCouche {}:", i);
            // On inverse l'ordre des rangées (7 down to 0) pour correspondre au 'reversed' Python
            for ligne in (0..8).rev() {
                // En Rust, {:?} est le formateur de debug pour afficher un tableau entier d'un coup
                println!("{:?}", tensor[i][ligne]);
            }
        }

        let fen = "rnbqkbnr/ppppp1p1/7p/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3";
        let tensor = fen_to_tensor(fen);
        // On suppose que 'tensor' est de type [[[u8; 8]; 8]; 19]
        println!("Aux blancs de jouer, prise en passant");

        for i in 0..19 {
            println!("\nCouche {}:", i);
            // On inverse l'ordre des rangées (7 down to 0) pour correspondre au 'reversed' Python
            for ligne in (0..8).rev() {
                // En Rust, {:?} est le formateur de debug pour afficher un tableau entier d'un coup
                println!("{:?}", tensor[i][ligne]);
            }
        }

        let fen = "rnbqkbnr/p2p1ppp/1p2p3/2p5/8/5NPB/PPPPPP1P/RNBQ1RK1 b kq - 1 4";
        let tensor = fen_to_tensor(fen);
        // On suppose que 'tensor' est de type [[[u8; 8]; 8]; 19]
        println!("Les blancs viennent de faire petit roque");

        for i in 0..19 {
            println!("\nCouche {}:", i);
            // On inverse l'ordre des rangées (7 down to 0) pour correspondre au 'reversed' Python
            for ligne in (0..8).rev() {
                // En Rust, {:?} est le formateur de debug pour afficher un tableau entier d'un coup
                println!("{:?}", tensor[i][ligne]);
            }
        }*/

        let fen = "rnbqkbnr/p3pppp/1p6/2p5/3p4/5NPB/PPPPPP1P/RNBQK2R b Qkq - 1 5";
        let tensor = fen_to_tensor(fen);
        // On suppose que 'tensor' est de type [[[u8; 8]; 8]; 19]
        println!("Les blancs ont bougé puis remis leur tour en place, le petit roque n'est plus possible");

        for i in 0..19 {
            println!("\nCouche {}:", i);
            // On inverse l'ordre des rangées (7 down to 0) pour correspondre au 'reversed' Python
            for ligne in (0..8).rev() {
                // En Rust, {:?} est le formateur de debug pour afficher un tableau entier d'un coup
                println!("{:?}", tensor[i][ligne]);
            }
        }
    }
}
