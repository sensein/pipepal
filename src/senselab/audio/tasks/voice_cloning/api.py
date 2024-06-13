"""This module implements some utilities for the voice cloning task."""

from typing import Any, Dict, List, Optional

import pydra

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import SenselabModel, TorchModel

from senselab.audio.tasks.voice_cloning.knnvc import KNNVC


def clone_voices(
    source_audios: List[Audio],
    target_audios: List[Audio],
    model: SenselabModel,
    device: Optional[DeviceType] = None,
    **kwargs: Dict[str, Any]
) -> List[Audio]:
    """Clones voices from source audios to target audios using the given model."""
    if len(source_audios) != len(target_audios):
        raise ValueError("Source and target audios must have the same length.")

    if isinstance(model, TorchModel) and model.path_or_uri == "bshall/knn-vc":
        topk = kwargs.get("topk", 4)
        if not isinstance(topk, int):
            raise ValueError("topk must be an integer.")
        prematched_vocoder = kwargs.get("prematched_vocoder", True)
        if not isinstance(prematched_vocoder, bool):
            raise ValueError("prematched_vocoder must be a boolean.")
        return KNNVC.clone_voices_with_knn_vc(
            source_audios=source_audios,
            target_audios=target_audios,
            model=model,
            prematched_vocoder=prematched_vocoder,
            topk=topk,
            device=device
        )
    else:
        raise NotImplementedError("Only KNNVC is supported for now.")

clone_voices_pt = pydra.mark.task(clone_voices)