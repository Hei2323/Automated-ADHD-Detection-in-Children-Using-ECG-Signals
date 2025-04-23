import torch
import torch.nn.functional as F
import numpy as np

class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hook to capture activations
        self.target_layer.register_forward_hook(self.save_activations)

    def save_activations(self, output):
        self.activations = output

    def forward(self, input_tensor, target_class):
        output = self.model(input_tensor)  # Forward pass

        # Compute scores for each convolutional channel
        activation_maps = self.activations.detach().cpu().numpy()  # (1, num_channels, length)
        scores = []

        for i in range(activation_maps.shape[1]):
            saliency_map = activation_maps[0, i, :]
            saliency_map = np.maximum(saliency_map, 0)  # Apply ReLU function

            # Interpolate to match input length
            saliency_map = np.interp(np.arange(input_tensor.shape[2]), np.arange(saliency_map.shape[0]), saliency_map)
            saliency_map = saliency_map - np.min(saliency_map)
            saliency_map = saliency_map / (np.max(saliency_map) + 1e-5)

            # Mask input and compute score
            masked_input = input_tensor.cpu().numpy() * saliency_map
            masked_input_tensor = torch.from_numpy(masked_input).to(input_tensor.device)
            score = F.softmax(self.model(masked_input_tensor.float())[0], dim=0)[target_class].item()
            scores.append(score)

        # Weighted sum to generate final Score-CAM map
        scores = np.array(scores)
        weighted_activations = np.zeros(activation_maps.shape[2])

        for i in range(activation_maps.shape[1]):
            weighted_activations += scores[i] * activation_maps[0, i, :]

        weighted_activations = np.maximum(weighted_activations, 0)
        weighted_activations = (weighted_activations - np.min(weighted_activations)) / (np.max(weighted_activations) + 1e-5)

        assert np.min(weighted_activations) >= 0, "Standardized activations contain negative values!"

        return weighted_activations
