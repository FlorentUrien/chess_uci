import torch
from onnx2torch import convert

class ONNXPredictor:
    def __init__(self, onnx_path):
        print(f"Chargement de l'accélérateur ONNX : {onnx_path}")
        # Chargement via la méthode validée
        self.model = convert(onnx_path)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.get_device_name(0) else "cpu")
        self.model.to(self.device)

        # L'ordre des sorties que nous avons validé ensemble
        self.output_names = ['check', 'heatmap', 'material', 'policy', 'progress', 'value']

    def predict(self, board_tensor):
        # On s'assure d'avoir la dimension batch (1, 19, 8, 8)
        if board_tensor.ndim == 3:
            board_tensor = board_tensor[None, ...]  # Équivalent à np.expand_dims

        with torch.no_grad():
            # Conversion en Tensor int8 (requis par ton graphe ONNX)
            input_tensor = torch.from_numpy(board_tensor.copy()).to(device=self.device, dtype=torch.int8)

            # Inférence
            outputs = self.model(input_tensor)

            # On retourne les résultats sous forme de numpy pour le MCTS
            # L'ordre [check, heatmap, material, policy, progress, value] est celui de Netron
            return {name: outputs[i].detach().cpu().numpy() for i, name in enumerate(self.output_names)}