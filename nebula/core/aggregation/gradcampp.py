import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine


class GradCAMPlusPlus:
    """Simple GradCAM++ implementation used for model comparison."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        input_tensor = input_tensor.requires_grad_(True)
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        loss = output[0, target_class]
        loss.backward(retain_graph=True)

        gradients = self.gradients[0]
        activations = self.activations[0]

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + activations.mul(gradients.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / (alpha_denom + 1e-8)

        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(0)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


class GradCamPPDefenseMixin:
    """Mixin adding GradCAM++ based model filtering before aggregation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        defense_cfg = self.config.participant["defense_args"]["gradcampp"]
        self._enabled = defense_cfg.get("enabled", False)
        self._threshold = defense_cfg.get("threshold", 0.5)
        self._samples = defense_cfg.get("samples", 5)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_target_layer(self, model: torch.nn.Module) -> torch.nn.Module:
        if hasattr(model, "features"):
            return model.features[-1]
        raise ValueError("Model does not have a 'features' attribute for GradCAM++")

    def _sample_images(self) -> list[torch.Tensor]:
        dataloader = self.engine.trainer.datamodule.bootstrap_dataloader()
        images = []
        for batch in dataloader:
            x, _ = batch
            for img in x:
                images.append(img.unsqueeze(0).to(self._device))
                if len(images) >= self._samples:
                    break
            if len(images) >= self._samples:
                break
        return images

    @staticmethod
    def _distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(cosine(vec1, vec2))

    def _filter_models(self, models: dict[str, tuple[dict, float]]) -> dict[str, tuple[dict, float]]:
        if not self._enabled:
            return models

        images = self._sample_images()
        if not images:
            logging.warning("GradCamPPDefense: no images sampled for defense.")
            return models

        trusted = {}
        ref_model = self.engine.trainer.model.to(self._device)
        ref_layer = self._get_target_layer(ref_model)
        cam_ref = GradCAMPlusPlus(ref_model, ref_layer)

        for peer, (state_dict, weight) in models.items():
            model_copy = copy.deepcopy(ref_model)
            model_copy.load_state_dict(state_dict)
            model_copy.to(self._device)
            cam_peer = GradCAMPlusPlus(model_copy, self._get_target_layer(model_copy))
            dists = []
            for img in images:
                heat_ref = cam_ref.generate(img)
                heat_peer = cam_peer.generate(img)
                dists.append(self._distance(heat_ref.flatten(), heat_peer.flatten()))
            avg_dist = float(np.mean(dists))
            if avg_dist <= self._threshold:
                trusted[peer] = (state_dict, weight)
            else:
                logging.info(f"GradCamPPDefense: model from {peer} flagged as malicious (distance={avg_dist:.4f})")
        return trusted

    def run_aggregation(self, models: dict[str, tuple[dict, float]]):
        models = self._filter_models(models)
        return super().run_aggregation(models)
