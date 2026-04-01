use super::from_real::real_to_ia;
use shakmaty::{Chess, Position};

pub fn coups_legaux(pos: &Chess) -> [u8; 584] {
    let mut legal_mask: [u8; 584] = [0u8; 584];
    let is_white = pos.turn().is_white();

    for mv in pos.legal_moves() {
        // On convertit le move interne en UciMove (type attendu par real_to_ia)
        let uci_move = mv.to_uci(shakmaty::CastlingMode::Standard);

        // Magie : real_to_ia(m: &UciMove, ...) devrait maintenant accepter &uci_move directement
        if let Ok(Some(idx)) = real_to_ia(&uci_move, is_white) {
            println!("coups_legaux : {} / {}", idx, uci_move);
            if idx < 4672 {
                legal_mask[idx / 8] |= 1 << (idx % 8);
            }
        }
    }

    println!("{} / {:08b}", legal_mask[0], legal_mask[0]);
    legal_mask
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {        
        for i in 0..20 {
            let mut legal_mask: [u8; 584] = [0u8; 584];
            legal_mask[i / 8] |= 1 << (i % 8);
            println!("{} -> {} / {:08b}", i, legal_mask[0], legal_mask[0]);
        }
    }
}
