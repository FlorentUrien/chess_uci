use shakmaty::{CastlingSide, Chess, Position, Role, fen::Fen, CastlingMode};

pub fn board_to_tensor(pos: &Chess) -> [[[u8; 8]; 8]; 19] {
    let mut tensor = [[[0u8; 8]; 8]; 19];
    let board = pos.board();
    let is_white = pos.turn().is_white();

    for sq in shakmaty::Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            let file = sq.file() as usize;
            let rank = sq.rank() as usize;
            // Mirror vertical si Noir joue
            let view_rank = if is_white { rank } else { 7 - rank };

            let piece_idx = match piece.role {
                Role::Pawn => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook => 3,
                Role::Queen => 4,
                Role::King => 5,
            };

            if piece.color == pos.turn() {
                tensor[piece_idx][view_rank][file] = 1;
                tensor[17][view_rank][file] = 1; // Amis
            } else {
                tensor[piece_idx + 6][view_rank][file] = 1;
                tensor[18][view_rank][file] = 1; // Ennemis
            }
        }
    }

    // Droits au roque
    let c = pos.castles();
    for r in 0..8 {
        for f in 0..8 {
            if c.has(pos.turn(), CastlingSide::KingSide) {
                tensor[12][r][f] = 1;
            }
            if c.has(pos.turn(), CastlingSide::QueenSide) {
                tensor[13][r][f] = 1;
            }
            if c.has(!pos.turn(), CastlingSide::KingSide) {
                tensor[14][r][f] = 1;
            }
            if c.has(!pos.turn(), CastlingSide::QueenSide) {
                tensor[15][r][f] = 1;
            }
        }
    }

    // En-passant
    if let Some(ep_sq) = pos.pseudo_legal_ep_square() {
        let file = ep_sq.file() as usize;
        let rank = ep_sq.rank() as usize;
        tensor[16][if is_white { rank } else { 7 - rank }][file] = 1;
    }

    tensor
}

/// Conversion d'un fen en tenseur 19x8x8.
/// Retourne un tableau 3D de 19 plans, 8 lignes, 8 colonnes.
pub fn fen_to_tensor(fen_str: &str) -> [[[u8; 8]; 8]; 19] {
    // 1. On transforme la chaîne FEN en un objet Board
    // .expect() fera crasher le moteur si la FEN est invalide (rare en UCI)
    let fen: Fen = fen_str.parse().expect("FEN syntaxiquement invalide");
    let pos: Chess = fen.into_position(CastlingMode::Standard).expect("Conversion fen -> chess invalide");

    // 2. On appelle la fonction de conversion
    board_to_tensor(&pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fen_to_tensor() {
        let fen = "rnbqkbnr/pppppppp/8/8/7P/8/PPPPPPP1/RNBQKBNR b KQkq - 0 1";
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
        }

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
