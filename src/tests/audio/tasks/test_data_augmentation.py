import pytest
from senselab.audio.tasks.data_augmentation import augment_hf_dataset_with_non_torch_aug
from tests.utils import get_mono_audio_test_dataset, get_stereo_audio_test_dataset
from audiomentations import Lambda, Compose
from senselab.utils.tasks.input_output import _from_dict_to_hf_dataset, _from_hf_dataset_to_dict
from datasets import Audio

def test_augment_hf_dataset_with_non_torch_aug_identity():
    audio_data = get_stereo_audio_test_dataset()

    def identity_aug(samples, sample_rate):
        return samples
    identity_composition = Compose([Lambda(transform=identity_aug, p=1.0)])
    
    
    dict_dataset = _from_hf_dataset_to_dict(audio_data)
    expected_output = dict_dataset['audio']
    

    test_output = augment_hf_dataset_with_non_torch_aug(
        dict_dataset, 
        identity_composition
    )
    print((expected_output[0].keys()))
    print((test_output['augmented_audio'][0].keys()))
    assert expected_output == test_output['augmented_audio']