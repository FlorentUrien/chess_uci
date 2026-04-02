use super::node::Node;
use crate::conv::from_ia::ia_to_real;
use shakmaty::{Chess, Move};
use std::fs::File;
use std::io::Write;

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
    ///
    /// # Retour
    ///
    pub fn backpropagate(&mut self, leaf_idx: usize, mut value: f32) {
        let mut current_idx = Some(leaf_idx);

        while let Some(idx) = current_idx {
            let node = &mut self.nodes[idx];

            // 1. Mise à jour du compteur de visite
            node.visit_count += 1;

            // 2. Mise à jour de la valeur
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
        m

        /* TODO: comprendre ce code bizarre alternatif
        root.children
            .iter()
            .max_by_key(|&(_mv, child_idx)| self.nodes[*child_idx].visit_count)
            .map(|(mv, _child_idx)| mv.clone())

        }*/
    }

    pub fn choisir_x_meilleurs_coups(&self, x: usize) -> Vec<Move> {
        let root = &self.nodes[0];

        // 1. On extrait les données en allant chercher DIRECTEMENT dans l'Arena via les child_idx
        let mut coups_stats: Vec<(&Move, u32, f32, f32)> = root
            .children
            .iter()
            .map(|(mv, child_idx)| {
                let enfant = &self.nodes[*child_idx]; // On va chercher le noeud enfant dans l'Arena
                let n = enfant.visit_count;
                let p = enfant.prior_p;
                let v = if n > 0 {
                    enfant.value_sum / n as f32
                } else {
                    0.0
                };
                (mv, n, v, p)
            })
            .collect();

        // 2. On trie par visites (Ordre décroissant)
        coups_stats.sort_by(|a, b| b.1.cmp(&a.1));

        // 3. Affichage pour debug
        println!("\n--- TOP x DES COUPS RÉELLEMENT EXPLORÉS ---");
        for (i, (mv, n, v, p)) in coups_stats.iter().take(x).enumerate() {
            println!(
                "{}. Coup: {:<6} | Visites: {:<8} | Score moy: {:.4} | Prior P: {:.4}",
                i + 1,
                mv.to_string(),
                n,
                v,
                p
            );
        }
        println!("--------------------------------\n");

        coups_stats
            .into_iter()
            .take(5)
            .map(|(m, _, _, _)| m.clone())
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

    pub fn export_dot_pro(&self, min_visits: u32) -> String {
        let mut dot = String::from(
            "digraph MCTS {\n  rankdir=LR;\n  node [shape=box, fontname=\"Arial\", style=filled];\n",
        );

        for (i, node) in self.nodes.iter().enumerate() {
            if node.visit_count >= min_visits {
                // Correction : pas de virgule traînante quand style est vide
                let fillcolor = if i == 0 {
                    "#d4edda".to_string()
                } else if node.value() > 0.0 {
                    "#cce5ff".to_string() // bleu clair = bon pour nous
                } else {
                    "#f8d7da".to_string() // rouge clair = mauvais
                };

                // Correction : pas de virgule dans le label (remplacée par espace)
                let label = format!("N{} V:{} Q:{:.2}", i, node.visit_count, node.value());

                dot.push_str(&format!(
                    "  {} [label=\"{}\", fillcolor=\"{}\"];\n",
                    i, label, fillcolor
                ));

                for (mv, child_idx) in &node.children {
                    if let Some(child_node) = self.nodes.get(*child_idx) {
                        if child_node.visit_count >= min_visits {
                            let ratio = child_node.visit_count as f32 / node.visit_count as f32;
                            let pen_width = (ratio * 5.0).max(0.5);

                            // Correction : les caractères spéciaux dans les labels d'arêtes
                            // doivent être entre guillemets (déjà le cas) mais sans virgule
                            dot.push_str(&format!(
                                "  {} -> {} [label=\"{}\" penwidth={:.1}];\n",
                                i, child_idx, mv, pen_width
                            ));
                        }
                    }
                }
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Pour obtenir une représentation graphique de l'arbre à l'aide par exemple de https://edotor.net/
    pub fn save_dot_to_file(&self, min_visits: u32, filename: &str) -> std::io::Result<()> {
        let content = self.export_dot_pro(min_visits);
        let mut file = File::create(filename)?;
        file.write_all(content.as_bytes())?;
        println!("Arbre sauvegardé dans : {}", filename);
        Ok(())
    }

    pub fn save_and_render(&self, min_visits: u32, filename: &str) -> std::io::Result<()> {
        // 1. Sauvegarde le .dot
        self.save_dot_to_file(min_visits, filename)?;

        // 2. Appelle graphviz pour générer un SVG
        let svg_name = filename.replace(".dot", ".svg");
        let output = std::process::Command::new("dot")
            .args(["-Tsvg", filename, "-o", &svg_name])
            .output();

        match output {
            Ok(o) if o.status.success() => {
                println!("SVG généré : {}", svg_name);
                // 3. Ouvre dans le navigateur
                std::process::Command::new("xdg-open")
                    .arg(&svg_name)
                    .spawn()
                    .ok();
            }
            Ok(o) => eprintln!("Erreur dot : {}", String::from_utf8_lossy(&o.stderr)),
            Err(_) => eprintln!("graphviz non installé — sudo dnf install graphviz"),
        }
        Ok(())
    }
}
