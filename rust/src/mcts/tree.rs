use super::node::Node;
use shakmaty::Move;

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
        coups_legaux: &[(Move, f32)], // Liste de (index_du_coup, probabilité_policy)
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

            // 5. On lie l'enfant au parent
            self.nodes[parent_idx].children.push((mv, child_idx));
        }

        // 6. On marque le parent comme étendu
        self.nodes[parent_idx].is_expanded = true;
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
}
