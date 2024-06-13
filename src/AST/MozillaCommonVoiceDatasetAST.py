import os
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchaudio.functional import pitch_shift, add_noise


class MozillaCommonVoiceDatasetAST(Dataset):
    def __init__(self, metadata, data_dir, sample_rate=16000, audio_length=6, is_augment=False):
        self.metadata = pd.read_csv(metadata)
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.is_augment = is_augment
        self.window_length = audio_length * sample_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, self.metadata.iloc[idx, 0])
        label = self.metadata.iloc[idx, 2]

        signal, sr = torchaudio.load(filepath)
        signal = signal.to(self.device)
        signal = self._convert_to_mono(signal)
        signal = self._augment_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._cut_or_pad(signal)

        return signal, label

    def _resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    @staticmethod
    def _convert_to_mono(signal):
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        return signal

    def _cut_or_pad(self, signal):
        # if signal is longer than window_length, pick a random window of length window_length
        if signal.shape[1] > self.window_length:
            start_idx = torch.randint(0, signal.shape[1] - self.window_length, (1,))
            signal = signal[:, start_idx:start_idx + self.window_length]
        # if signal is shorter than window_length, pad with zeros
        else:
            pad = torch.zeros(1, self.window_length - signal.shape[1]).to(self.device)
            signal = torch.cat([signal, pad], dim=1)
        return signal

    def _augment_if_necessary(self, signal):
        if self.is_augment:
            random_pitch_shift = torch.randint(-5, 5, (1,)).item()
            noise = torch.randn_like(signal).to(self.device)
            snr = torch.randint(20, 30, (1,)).to(self.device)

            signal = pitch_shift(signal, self.sample_rate, n_steps=random_pitch_shift)
            signal = add_noise(signal, noise, snr=snr)
        return signal


def get_dataloader(metadata, data_dir, is_augment=False, batch_size=16, num_workers=0):
    dataset = MozillaCommonVoiceDatasetAST(metadata=metadata, data_dir=data_dir, is_augment=is_augment)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_labels(metadata):
    metadata = pd.read_csv(metadata)
    return dict(zip(metadata['accents'], metadata['accent']))


def ids2labels(labels):
    return {v: k for k, v in labels.items()}


def get_num_labels(metadata):
    metadata = pd.read_csv(metadata)
    return len(metadata['accents'].unique())


def get_classes(metadata_path):
    df = pd.read_csv(metadata_path)
    # get one unique value for each class
    dic = dict(zip(df['accent'], df['accents']))
    # return list ordered by the class number
    return [dic[i] for i in range(len(dic))]


def main():
    metadata = r"../../data/metadata/train.csv"
    data_dir = r"../../data/clips/"

    dataloader = get_dataloader(metadata, data_dir)

    for batch in dataloader:
        print(batch[0].shape, batch[1])
        break


if __name__ == "__main__":
    main()
