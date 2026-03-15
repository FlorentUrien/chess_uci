use shakmaty::Move;

pub struct Node {
    // Liens vers l'Arena (au lieu de pointeurs objets)
    pub parent: Option<usize>,        // Index du parent dans le Vec (8o)
    pub children: Vec<(Move, usize)>, // Liste de (Move (16o), index_noeud_enfant (8o) = 24o)

    // Statistiques MCTS (identiques à ton code Python)
    pub visit_count: u32, // u32 suffit largement pour des simulations (4o)
    pub value_sum: f32,   // Somme des valeurs (Q-value) (4o)
    pub prior_p: f32,     // Probabilité issue de la policy (4o)

    // États
    pub is_expanded: bool, // Déjà exploré ? (1o)
    pub is_pending: bool,  // En cours d'exploration ? (1o)
}

impl Node {
    /// Crée un nouveau nœud (similaire au __init__ Python)
    pub fn new(parent: Option<usize>, prior_p: f32) -> Self {
        Self {
            parent,
            children: Vec::new(),
            visit_count: 0,
            value_sum: 0.0,
            prior_p,
            is_expanded: false,
            is_pending: false,
        }
    }

    /// Calcul de la valeur moyenne (similaire à la property value)
    pub fn value(&self) -> f32 {
        if self.visit_count == 0 {
            return 0.0;
        }
        self.value_sum / (self.visit_count as f32)
    }

    /// Sélectionne le meilleur coup et l'index de l'enfant correspondant selon la formule PUCT.
    ///
    /// # Arguments
    /// * `nodes` - L'arbre que l'on va développer
    /// * `cpuct` - Le CPUCT
    ///
    /// # Retour
    /// Retourne (Le coup de transition vers le noeud enfant, index_du_noeud_enfant).
    pub fn select_child(&self, nodes: &[Node], cpuct: f32) -> Option<(Move, usize)> {
        if self.children.is_empty() {
            return None;
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut best_choice = None;

        // On pré-calcule la racine du nombre de visites pour gagner du temps
        let parent_visits = self.visit_count as f32;
        // On utilise max(1.0) pour s'assurer que l'exploration n'est pas nulle au premier coup
        let sqrt_total_visits = parent_visits.max(1.0).sqrt();

        for (mv, child_idx) in &self.children {
            let child = &nodes[*child_idx]; // Accès sécurisé via l'Arena

            // Calcul de l'exploration (U-score)
            // Formule : cpuct * P(s,a) * sqrt(Sum(N)) / (1 + N(s,a))
            let u_score =
                cpuct * child.prior_p * sqrt_total_visits / (1.0 + child.visit_count as f32);

            let score = -child.value() + u_score;

            if score > best_score {
                best_score = score;
                best_choice = Some((mv.clone(), *child_idx));
            }
        }

        best_choice
    }
}
