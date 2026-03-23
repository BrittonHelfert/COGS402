"""SAE utilities: loading and projection operations.

Requires the `dictionary_learning` package (installed as part of diffing-toolkit deps).
"""

import os
import torch
import einops
from pathlib import Path
from typing import Any
from huggingface_hub import snapshot_download
from dictionary_learning.utils import load_dictionary


def load_sae_from_path(sae_path: str, device: str = "cpu"):
    """Load SAE from a local path or HuggingFace repo subfolder."""
    local_path = Path(sae_path)
    if local_path.exists() and local_path.is_dir():
        final_path = str(local_path)
    else:
        parts = sae_path.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid path: {sae_path}")
        repo_id = f"{parts[0]}/{parts[1]}"
        subfolder = "/".join(parts[2:]) if len(parts) > 2 else None
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{subfolder}/*" if subfolder else None,
            local_dir_use_symlinks=False,
        )
        final_path = os.path.join(local_dir, subfolder) if subfolder else local_dir

    sae, _ = load_dictionary(final_path, device=device)
    sae.eval()
    return sae


def load_sae_for_layer(sae_repo: str, layer: int, trainer: int = 1, device: str = "cuda"):
    """Load SAE for a specific layer from a HuggingFace repo.

    Assumes the repo structure: {sae_repo}/resid_post_layer_{layer}/trainer_{trainer}
    """
    path = f"{sae_repo}/resid_post_layer_{layer}/trainer_{trainer}"
    return load_sae_from_path(path, device=device)


def compute_feature_cosine_similarities(
    activation_diff: torch.Tensor,
    sae: Any,
    device: str = "cuda",
) -> torch.Tensor:
    """Cosine similarity between a difference vector and all SAE feature directions.

    Returns a tensor of shape (n_features,).
    """
    decoder = sae.decoder.weight
    # decoder shape is either (hidden, n_features) or (n_features, hidden)
    features = decoder.T if decoder.shape[0] == activation_diff.shape[0] else decoder

    diff_norm = (activation_diff / activation_diff.norm()).to(torch.float32).to(device)
    feat_norm = (features / features.norm(dim=1, keepdim=True)).to(torch.float32).to(device)
    return einops.einsum(feat_norm, diff_norm, "f d, d -> f")


def project_onto_feature(
    activations: torch.Tensor,
    feature_idx: int,
    sae: Any,
    device: str = "cuda",
) -> torch.Tensor:
    """Project activations onto a single SAE feature decoder direction.

    Returns a tensor of shape (batch,).
    """
    decoder = sae.decoder.weight
    if decoder.shape[0] == activations.shape[-1]:
        direction = decoder[:, feature_idx]
    else:
        direction = decoder[feature_idx, :]
    direction_norm = (direction / direction.norm()).to(device)
    return einops.einsum(activations.to(device), direction_norm, "b d, d -> b").cpu()
