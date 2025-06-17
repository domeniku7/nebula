import copy
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from collections import defaultdict


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
        """Return the last convolutional layer of ``model``.

        GradCAM++ requires a convolutional layer to register the hooks used to
        compute the saliency maps.  Different architectures name their feature
        extraction blocks differently (``features`` in some torchvision models,
        ``conv1``/``conv2``/... in others).  This helper searches for the last
        :class:`torch.nn.Conv2d` instance in the model so that it works across
        the various models shipped with NEBULA.
        """

        # If the model exposes a ``features`` attribute (e.g. VGG-like models),
        # try to obtain the last convolution from it first.
        if hasattr(model, "features"):
            features = getattr(model, "features")
            conv_layers = [m for m in features.modules() if isinstance(m, torch.nn.Conv2d)]
            if conv_layers:
                return conv_layers[-1]
            # Fallback: if ``features`` behaves like a sequence or mapping, pick
            # its last module.
            if isinstance(features, (torch.nn.Sequential, torch.nn.ModuleList)):
                return features[-1]
            if isinstance(features, torch.nn.ModuleDict) and features:
                return list(features.values())[-1]

        # Generic fallback: search the whole model for convolutional layers.
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        if conv_layers:
            return conv_layers[-1]

        raise ValueError(
            "Could not locate a convolutional layer for GradCAM++"
        )

    def _sample_images(self) -> list[torch.Tensor]:
        dataloader = self.engine.trainer.datamodule.bootstrap_dataloader()
        class_to_images = defaultdict(list)

        # Accumulate images by class (up to 5 per class)
        for batch in dataloader:
            x, y = batch
            for img, label in zip(x, y):
                class_id = int(label)
                if len(class_to_images[class_id]) < 5:
                    class_to_images[class_id].append(img.unsqueeze(0).to(self._device))

            if len(class_to_images) >= 5 and all(len(v) >= 1 for v in class_to_images.values()):
                break

        # Sort classes by how many images they have, descending
        sorted_classes = sorted(class_to_images.items(), key=lambda x: len(x[1]), reverse=True)
        selected_classes = sorted_classes[:5]  # Choose up to 5 best-filled classes

        if not selected_classes:
            logging.warning("GradCamPPDefense: No eligible classes found for sampling.")
            return []

        selected_images = []
        for _, imgs in selected_classes:
            selected_images.extend(random.sample(imgs, k=min(5, len(imgs))))

        return selected_images

    @staticmethod
    def _distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Normalize vectors to [0, 1]
        vec1 = (vec1 - vec1.min()) / (vec1.max() - vec1.min() + 1e-8)
        vec2 = (vec2 - vec2.min()) / (vec2.max() - vec2.min() + 1e-8)

        # Handle NaN or zero-norm cases
        if np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)):
            logging.warning("GradCamPPDefense: NaN detected in saliency vectors.")
            return None
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            logging.warning("GradCamPPDefense: zero vector detected in saliency vectors.")
            return None

        return float(cosine(vec1, vec2))

    def _filter_models(self, models: dict[str, tuple[dict, float]]) -> dict[str, tuple[dict, float]]:
        if not self._enabled:
            return models

        images = self._sample_images()
        if not images:
            logging.warning("GradCamPPDefense: no images sampled for defense.")
            return models

        trusted = {}
        peer_distances_cache = {}

        local_addr = self._addr
        local_model = models.get(local_addr)

        ref_model = self.engine.trainer.model.to(self._device)
        ref_layer = self._get_target_layer(ref_model)
        cam_ref = GradCAMPlusPlus(ref_model, ref_layer)

        # Compute distances for each peer model
        for peer, (state_dict, _weight) in models.items():
            if peer == self._addr:
                # Skip comparison with the local model to avoid skewing the
                # dynamic threshold toward zero.
                continue
            model_copy = copy.deepcopy(ref_model)
            model_copy.load_state_dict(state_dict)
            model_copy.to(self._device)
            cam_peer = GradCAMPlusPlus(model_copy, self._get_target_layer(model_copy))

            dists = []
            for img in images:
                heat_ref = cam_ref.generate(img)
                heat_peer = cam_peer.generate(img)
                vec1 = heat_ref.flatten()
                vec2 = heat_peer.flatten()
                dist = self._distance(vec1, vec2)
                if dist is not None:
                    dists.append(dist)

            avg_dist = float(np.mean(dists))
            peer_distances_cache[peer] = avg_dist

        # Calculate dynamic threshold ignoring infinite distances
        all_peer_distances = list(peer_distances_cache.values())
        finite_dists = [d for d in all_peer_distances if np.isfinite(d)]
        if finite_dists:
            threshold = float(np.mean(finite_dists))
        else:
            logging.warning("GradCamPPDefense: no valid distances, using configured threshold")
            threshold = self._threshold
        logging.info(f"GradCamPPDefense: Dynamic threshold = {threshold:.4f}")

        # Filter based on threshold and distance validity
        for peer, avg_dist in peer_distances_cache.items():
            logging.info(f"GradCamPPDefense: Peer {peer} distance to local model: {avg_dist:.4f}")
            if not np.isfinite(avg_dist):
                logging.info(f"GradCamPPDefense: model from {peer} flagged as malicious (invalid distance)")
                continue
            if avg_dist <= threshold:
                trusted[peer] = models[peer]
            else:
                logging.info(f"GradCamPPDefense: model from {peer} flagged as malicious (distance={avg_dist:.4f})")
        if local_model is not None:
            # Always include the local model in the aggregation set
            trusted[local_addr] = local_model

        return trusted

    def run_aggregation(self, models: dict[str, tuple[dict, float]]):
        models = self._filter_models(models)
        return super().run_aggregation(models)
