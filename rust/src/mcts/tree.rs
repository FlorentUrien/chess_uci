use super::node::Node;
use crate::conv::from_ia::ia_to_real;
use shakmaty::{Chess, Move};

pub struct MctsTree {
    pub nodes: Vec<Node>,
    pub profondeur: u8,
}

impl MctsTree {
    pub fn new(root_node: Node) -> Self {
        Self {
            nodes: vec![root_node],
            profondeur: 0,
        }
    }

    /// Étend un nœud en lui ajoutant des enfants pour chaque coup légal
    ///
    /// # Arguments
    /// * `parent_idx` - L'index du nœud que l'on étend
    /// * `coups_legaux` - La liste des coups légaux et leurs probabilités
    ///
    /// # Retour
    ///
    pub fn expand_node(
        &mut self,
        parent_idx: usize,
        coups_legaux: &[(usize, f32)],
        board: &Chess, // Liste de (index_du_coup, probabilité_policy)
    ) {
        // 1. On vérifie si le nœud n'est pas déjà étendu (sécurité)
        if self.nodes[parent_idx].is_expanded {
            return;
        }

        // 2. Pour chaque coup fourni par ton modèle ONNX/moteur
        for &(mv, proba) in coups_legaux {
            let child_idx = self.nodes.len(); // L'index du futur enfant est la taille actuelle du Vec

            // 3. Création du nouveau nœud
            let new_child = Node::new(Some(parent_idx), proba);

            // 4. On l'ajoute à l'Arena
            self.nodes.push(new_child);

            // 5. On récupère le UciMove depuis ton index IA
            let uci_m = ia_to_real(mv, board).expect("Index IA invalide");

            // 6. On le convertit en Move réel par rapport au board actuel
            let m = uci_m
                .to_move(board)
                .expect("Coup UCI illégal pour cette position");

            // 7. On lie l'enfant au parent
            self.nodes[parent_idx].children.push((m, child_idx));
        }

        // 6. On marque le parent comme étendu
        self.nodes[parent_idx].is_expanded = true;
        self.nodes[parent_idx].is_pending = false;
    }

    /// Remonte les résultats de l'évaluation jusqu'à la racine
    ///
    /// # Arguments
    /// * `leaf_idx` - index du nœud à partir duquel on va remonter
    /// * `value` - La valeur de ce nœud
    /// * `virtual_loss` - La virtual loss qu'il ne faut pas oublier de décrémenter
    ///
    /// # Retour
    ///
    pub fn backpropagate(&mut self, leaf_idx: usize, mut value: f32, virtual_loss: u32) {
        let mut current_idx = Some(leaf_idx);

        while let Some(idx) = current_idx {
            let node = &mut self.nodes[idx];

            // 1. On retire la perte virtuelle (ex: 10)
            // 2. On ajoute la visite réelle (+1)
            // Bilan net : -9 si virtual_loss était de 10
            node.visit_count = node.visit_count.saturating_sub(virtual_loss) + 1;

            // 3. Mise à jour de la valeur
            node.value_sum += value;

            // 4. On remonte au parent
            current_idx = node.parent;

            // 5. Inversion de la perspective (Minimax)
            value = -value;
        }
    }

    /// Sélectionne le meilleur coup à jouer à partir de la racine.
    /// Retourne le coup ayant le plus grand nombre de visites.
    pub fn choisir_meilleur_coup(&self) -> Option<Move> {
        // La racine est toujours le premier nœud de notre Arena.
        let root = &self.nodes[0];

        // On cherche parmi les enfants de la racine celui qui a le visit_count le plus élevé.
        let mut nb = 0;
        let mut m = None;
        for n in &root.children {
            if self.nodes[n.1].visit_count >= nb {
                nb = self.nodes[n.1].visit_count;
                m = Some(n.0.clone());
            }
        }
        println!("Nb = {}", nb);
        m

        /*root.children
            .iter()
            .max_by_key(|&(_mv, child_idx)| self.nodes[*child_idx].visit_count)
            .map(|(mv, _child_idx)| mv.clone())

        }*/
    }

    pub fn choisir_5_meilleurs_coups(&self) -> Vec<Move> {
        let root = &self.nodes[0];

        // 1. On extrait les données en allant chercher DIRECTEMENT dans l'Arena via les child_idx
        let mut coups_stats: Vec<(&Move, u32, f32)> = root
            .children
            .iter()
            .map(|(mv, child_idx)| {
                let enfant = &self.nodes[*child_idx]; // On va chercher le noeud enfant dans l'Arena
                let n = enfant.visit_count;
                let v = if n > 0 {
                    enfant.value_sum / n as f32
                } else {
                    0.0
                };
                (mv, n, v)
            })
            .collect();

        // 2. On trie par visites (Ordre décroissant)
        coups_stats.sort_by(|a, b| b.1.cmp(&a.1));

        // 3. Affichage pour debug
        println!("\n--- TOP 5 DES COUPS RÉELLEMENT EXPLORÉS ---");
        for (i, (mv, n, v)) in coups_stats.iter().take(5).enumerate() {
            println!(
                "{}. Coup: {:<6} | Visites: {:<8} | Score moy: {:.4}",
                i + 1,
                mv.to_string(),
                n,
                v
            );
        }
        println!("--------------------------------\n");

        coups_stats
            .into_iter()
            .take(5)
            .map(|(m, _, _)| m.clone())
            .collect()
    }

    pub fn affiche(&self) {
        println!("Profondeur de l'arbre = {}", self.profondeur);
        println!("Nombre de noeuds = {}", self.nodes.len());
        self.ss_arbre(0);
    }

    fn ss_arbre(&self, racine: usize) {
        if self.nodes[racine].is_pending {
            println!(" <is pending>");
        }
        if self.nodes[racine].is_expanded {
            print!("{} = {}", racine, self.nodes[racine].value_sum);

            println!("---------------------------------------");
            for (move_obj, node_idx) in &self.nodes[racine].children {
                println!("Coup possible : {} -> {} : {}", racine, node_idx, move_obj);
                self.ss_arbre(*node_idx);
            }
            println!("---------------------------------------");
        }
    }

    pub fn affiche_sonnet(&self) {
        println!(
            "Profondeur : {}  |  Nœuds : {}",
            self.profondeur,
            self.nodes.len()
        );
        // self.affiche_noeud_sonnet(0, 0, None);
    }

    fn affiche_noeud_sonnet(&self, idx: usize, depth: usize, move_label: Option<&str>) {
        let node = &self.nodes[idx];
        let indent = "  ".repeat(depth);
        let connector = if depth == 0 { "" } else { "└ " };

        let val_str = format!("{:+.4}", node.value_sum);

        let pending = if node.is_pending { " ⏳" } else { "" };
        let children_hint = if !node.children.is_empty() {
            format!(" ({} fils)", node.children.len())
        } else {
            String::new()
        };

        println!(
            "{}{}{:?}#{:<5} val={} N=VC{}P{}CH{}",
            indent, connector, move_label, idx, val_str, node.visit_count, pending, children_hint
        );

        for (mv, child_idx) in &node.children {
            let mv_str = format!("{}", mv);
            self.affiche_noeud_sonnet(*child_idx, depth + 1, Some(&mv_str));
        }
    }
}
