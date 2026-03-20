from shared_interface import SharedInterface
# from predictor import ONNXPredictor

def main():
    # 1. On initialise l'interface (c'est Python qui 'create=True' les segments)
    # Assure-toi que le batch_size correspond à celui de ton code Rust
    interface = SharedInterface(batch_size=512, create=True)

    try:
        while True:
            # 3. On attend que Rust dise "J'ai fini d'écrire les positions"
            # Cette fonction bloque jusqu'à ce que sync[0] != 0
            interface.wait_for_input()
            
            # 4. Récupération des données depuis la mémoire partagée
            # L'interface te donne déjà une vue Numpy (self.tensor)
            # input_tensor = interface.tensor 
            
            # 5. Inférence (Le moment où le GPU travaille)
            # Predictor retourne un dict avec 'policy', 'value', etc.
            # predictions = predictor.predict(input_tensor)
            
            # 6. On recopie les résultats dans la mémoire partagée pour Rust
            # On utilise [:] pour modifier le contenu du segment sans casser la vue
            # interface.policy[:] = predictions['policy']
            # interface.value[:] = predictions['value']
            
            # 7. On signale à Rust que c'est prêt
            # On lui renvoie la taille du batch pour qu'il sache combien lire
            # interface.signal_output_ready(batch_size=512)

    except KeyboardInterrupt:
        print("\nArrêt de l'IA...")
    finally:
        interface.close()

if __name__ == "__main__":
    main()