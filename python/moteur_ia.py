import sys

from predictor import ONNXPredictor


def charger_modele(model_path: str) -> ONNXPredictor:
    """
    On charge le modèle IA

    :param model_path: Le modèle ONNX a charger
    :return: Le modèle chargé
    """
    print(f"info string Chargement du Modèle {model_path}", flush=True)

    try:
        model = ONNXPredictor(model_path + ".onnx")

    except Exception as e:
        print(f"info string Erreur fatale : {str(e)}", flush=True)
        return None

    if model is None:
        print(f"info string Le chargement du Modèle {model_path} a échoué", flush=True)
    else:
        print(f"info string Le chargement du Modèle {model_path} a réussi", flush=True)

    return model
