"""This module implements some utilities to extract speaker embeddings from a model."""

from typing import List, Optional

import pydra
import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_embeddings.speechbrain import SpeechBrainEmbeddings
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel, SenselabModel


def extract_speaker_embeddings_from_audios(
    audios: List[Audio],
    model: SenselabModel = HFModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main"),
    device: Optional[DeviceType] = None,
) -> List[torch.Tensor]:
    """Compute the speaker embedding of audio signals.

    Args:
        audios (List[Audio]): A list of Audio objects containing the audio signals and their properties.
        model (SenselabModel): The model used to compute the embeddings
            (default is "speechbrain/spkrec-ecapa-voxceleb").
        device (Optional[DeviceType]): The device to run the model on (default is None).

    Returns:
        List[torch.Tensor]: A list of 1d tensors containing the speaker embeddings for each audio file.

    Raises:
        NotImplementedError: If the model is not a Hugging Face model.

    Examples:
        >>> audios = [Audio.from_filepath("sample.wav")]
        >>> model = HFModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")
        >>> embeddings = extract_speaker_embeddings_from_audios(audio, model, device=DeviceType.CUDA)
        >>> print(embeddings[0].shape)
        torch.Size([192])
    """
    if isinstance(model, HFModel):  # TODO: check that this is a speechbrain model!
        return SpeechBrainEmbeddings.extract_speechbrain_speaker_embeddings_from_audios(
            audios=audios, model=model, device=device
        )
    else:
        raise NotImplementedError("The specified model is not supported for now.")


extract_speaker_embeddings_from_audios_pt = pydra.mark.task(extract_speaker_embeddings_from_audios)
