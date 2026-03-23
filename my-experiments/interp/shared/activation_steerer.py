"""Hook-based activation steering: addition and ablation of directions."""

import torch
import einops
from typing import Union, Iterable, List


class ActivationIntervention:
    def __init__(
        self,
        vector: torch.Tensor,
        coefficient: float,
        layer_indices: Union[int, List[int]],
        intervention_type: str = "ablation",
    ):
        """
        Args:
            vector: The direction to steer (hidden_size,).
            coefficient: For ablation, the replacement scale (0 = full ablation).
                         For addition, the steering multiplier.
            layer_indices: Which layers to apply this intervention to.
            intervention_type: 'ablation' or 'addition'.
        """
        self.vector = vector
        self.coefficient = float(coefficient)
        self.layer_indices = [layer_indices] if isinstance(layer_indices, int) else list(layer_indices)
        self.intervention_type = intervention_type.lower()
        if self.intervention_type not in {"addition", "ablation"}:
            raise ValueError(f"intervention_type must be 'addition' or 'ablation', got '{intervention_type}'")


class ActivationSteerer:
    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "model.layers",
        "model.model.layers",
    )

    def __init__(
        self,
        model: torch.nn.Module,
        interventions: List[ActivationIntervention],
        *,
        positions: str = "all",
    ):
        """
        Args:
            model: The transformer model to hook.
            interventions: List of ActivationIntervention objects.
            positions: 'all' to intervene on every token, 'last' for last token only.
        """
        self.model = model
        self.positions = positions.lower()
        self._handles = []
        if self.positions not in {"all", "last"}:
            raise ValueError("positions must be 'all' or 'last'")

        p = next(self.model.parameters())
        hidden_size = getattr(self.model.config, "hidden_size", None)
        self.interventions_by_layer: dict[int, list] = {}

        for iv in interventions:
            vec = torch.as_tensor(iv.vector, dtype=p.dtype, device=p.device)
            if vec.ndim != 1:
                raise ValueError(f"Vector must be 1-D, got {vec.shape}")
            if hidden_size and vec.numel() != hidden_size:
                raise ValueError(f"Vector length {vec.numel()} != hidden_size {hidden_size}")
            for layer_idx in iv.layer_indices:
                self.interventions_by_layer.setdefault(layer_idx, []).append(
                    (vec, iv.coefficient, iv.intervention_type)
                )

    def _locate_layer_list(self):
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if hasattr(cur, "__getitem__"):
                    return cur
        raise ValueError("Cannot find layer list. Add the attr path to _POSSIBLE_LAYER_ATTRS.")

    def _create_hook(self, layer_idx: int):
        def hook(module, ins, out):
            layers = out if torch.is_tensor(out) else out[0]
            was_tuple = not torch.is_tensor(out)
            modified = layers
            for vec, coeff, itype in self.interventions_by_layer[layer_idx]:
                if itype == "addition":
                    modified = self._add(modified, vec, coeff)
                else:
                    modified = self._ablate(modified, vec, coeff)
            return (modified, *out[1:]) if was_tuple else modified
        return hook

    def _add(self, acts, vec, coeff):
        steer = coeff * vec
        if self.positions == "all":
            return acts + steer
        result = acts.clone()
        result[:, -1, :] += steer
        return result

    def _ablate(self, acts, vec, coeff):
        vn = vec / (vec.norm() + 1e-8)
        if self.positions == "all":
            proj = einops.einsum(acts, vn, "b l d, d -> b l")
            return acts - einops.einsum(proj, vn, "b l, d -> b l d") + coeff * vec
        result = acts.clone()
        last = result[:, -1, :]
        proj = einops.einsum(last, vn, "b d, d -> b")
        result[:, -1, :] = last - einops.einsum(proj, vn, "b, d -> b d") + coeff * vec
        return result

    def __enter__(self):
        layers = self._locate_layer_list()
        for layer_idx in self.interventions_by_layer:
            handle = layers[layer_idx].register_forward_hook(self._create_hook(layer_idx))
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        self.remove()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
