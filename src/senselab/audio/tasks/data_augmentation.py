"""This module implements some utilities for the audio data augmentation task."""
from typing import Any, Dict

import torch
import numpy as np
from datasets import Dataset
from torch_audiomentations import Compose
from audiomentations import Compose as NonTorchCompose

from senselab.utils.tasks.input_output import _from_dict_to_hf_dataset, _from_hf_dataset_to_dict


def augment_hf_dataset(dataset: Dict[str, Any], augmentation: Compose, audio_column: str = 'audio') -> Dict[str, Any]:
    """Resamples a Hugging Face `Dataset` object."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)

    def _augment_hf_row(row: Dataset, augmentation: Compose, audio_column: str) -> Dict[str, Any]:
        waveform = row[audio_column]['array']
        sampling_rate = row[audio_column]['sampling_rate']

        # Ensure waveform is a PyTorch tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [num_samples] -> [1, 1, num_samples]
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [batch_size, num_samples] -> [batch_size, 1, num_samples]

        augmented_hf_row = augmentation(waveform, sample_rate=sampling_rate).squeeze()

        return { "augmented_audio": {
                    "array": augmented_hf_row,
                    "sampling_rate": sampling_rate 
                    }
            }

    augmented_hf_dataset = hf_dataset.map(lambda x: _augment_hf_row(x, augmentation, audio_column))
    augmented_hf_dataset = augmented_hf_dataset.remove_columns([audio_column])
    return _from_hf_dataset_to_dict(augmented_hf_dataset)


def augment_hf_dataset_with_non_torch_aug(dataset: Dict[str, Any], augmentation: NonTorchCompose, audio_column: str = 'audio') -> Dict[str, Any]:
    """Resamples a Hugging Face `Dataset` object."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)

    def _augment_hf_row(row: Dataset, augmentation: NonTorchCompose, audio_column: str) -> Dict[str, Any]:
        waveform = row[audio_column]['array']
        sampling_rate = row[audio_column]['sampling_rate']

        # Ensure waveform is a NumPy array
        if not isinstance(waveform, np.ndarray):
            waveform = np.array(waveform)
        if len(waveform.shape) == 1:
            waveform = np.expand_dims(np.expand_dims(waveform,0),0)  # [num_samples] -> [1, 1, num_samples]
        elif len(waveform.shape) == 2:
            waveform = np.expand_dims(waveform,1)  # [batch_size, num_samples] -> [batch_size, 1, num_samples]

        augmented_hf_row = augmentation(waveform, sample_rate=sampling_rate).squeeze()
        
        #convert to Tensor for internal consistency
        return { "augmented_audio": {
                    "array": torch.tensor(augmented_hf_row),
                    "sampling_rate": sampling_rate 
                    }
            }

    augmented_hf_dataset = hf_dataset.map(lambda x: _augment_hf_row(x, augmentation, audio_column))
    print(augmented_hf_dataset)
    augmented_hf_dataset = augmented_hf_dataset.remove_columns([audio_column])
    return _from_hf_dataset_to_dict(augmented_hf_dataset)


