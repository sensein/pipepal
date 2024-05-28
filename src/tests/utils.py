from datasets import Dataset
import torchaudio

def get_test_output_dataset():
    return Dataset.from_file('src/tests/data_for_testing/output_dataset/data-00000-of-00001.arrow')

def get_mono_audio_test_dataset():
    waveform, sr = torchaudio.load('src/tests/data_for_testing/audio_48khz_mono_16bits.wav')
    dataset_dict = {
        'audio': [{
            'array': waveform.squeeze(),
            'sampling_rate': sr,
            # 'path': 'src/tests/data_for_testing/audio_48khz_mono_16bits.wav'
        }],
    }
    return Dataset.from_dict(dataset_dict)


def get_stereo_audio_test_dataset():
    waveform, sr = torchaudio.load('src/tests/data_for_testing/audio_48khz_stereo_16bits.wav')
    print(waveform.shape)
    dataset_dict = {
        'audio': [{
            'array': waveform,
            'sampling_rate': sr,
            # 'path': 'src/tests/data_for_testing/audio_48khz_stereo_16bits.wav'
        }],
    }
    return Dataset.from_dict(dataset_dict)
