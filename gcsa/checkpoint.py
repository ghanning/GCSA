import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .model import GCSA


def save_checkpoint(model: GCSA, model_args: Dict, affinity: List[str], path: Path) -> None:
    """! Save checkpoint.

    @param model GCSA model.
    @param model_args Arguments used to create the model.
    @param affinity The types of affinity used.
    @param path Checkpoint path.
    """
    state = model_args.copy()
    state["model"] = model.state_dict()
    state["affinity"] = affinity
    logging.info(f"Saving checkpoint to {path}")
    torch.save(state, path)


def load_checkpoint(path: Path, device: str) -> Tuple[GCSA, List[str]]:
    """! Load checkpoint and create model.

    @param path Checkpoint path.
    @param device Target device.
    @return The resulting GCSA model and the list of affinity types.
    """
    logging.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    params = {k: v for k, v in checkpoint.items() if k != "model"}
    affinity = params.pop("affinity")
    logging.debug(f"{params=} {affinity=}")

    model = GCSA(**params)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    return model, affinity
