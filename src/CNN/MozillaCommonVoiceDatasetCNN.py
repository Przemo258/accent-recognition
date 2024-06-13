import torch
import torchaudio
import pandas as pd
import os

from torch.utils.data import Dataset
from torchaudio import transforms
from torchaudio.functional import pitch_shift, add_noise
from torch.utils.data import DataLoader


def main():
    metadata = r"../../data/metadata/train.csv"
    data_path = r"../../data/clips/"

    dataset = MozillaCommonVoiceDatasetCNN(metadata, data_path)
    item, label = dataset[0]
    print(item.shape, label)
    print(get_num_classes(metadata))


class MozillaCommonVoiceDatasetCNN(Dataset):
    def __init__(self, metadata, data_dir, sample_rate=16000,
                 n_mels=128, device="cpu", augment=False, audio_length=6):
        self.metadata = pd.read_csv(metadata)
        self.data_dir = data_dir
        self.target_sample_rate = sample_rate
        self.n_mels = n_mels
        self.window_length = audio_length * sample_rate
        self.device = device
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)

        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_or_pad(signal)

        signal = self._augment_audio_if_necessary(signal)

        mel = self._signal_to_mel_spectrogram(signal)
        mel = self._augment_spectrogram_if_necessary(mel)

        return mel, label

    def _signal_to_mel_spectrogram(self, signal):
        mel_transformer = transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_mels=self.n_mels, n_fft=1024, hop_length=512)
        amp_to_db = transforms.AmplitudeToDB()

        mel_transformer = mel_transformer.to(self.device)
        amp_to_db = amp_to_db.to(self.device)

        mel = mel_transformer(signal)
        mel = amp_to_db(mel)

        return mel

    def _get_audio_sample_path(self, idx):
        return os.path.join(self.data_dir, self.metadata.iloc[idx, 0])

    def _get_audio_sample_label(self, idx):
        return self.metadata.iloc[idx, 2]

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            resampler = resampler.to(self.device)
            signal = resampler(signal)
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

    def _augment_audio_if_necessary(self, signal):
        if not self.augment:
            return signal

        pitch_steps = torch.randint(-5, 5, (1,)).item()
        signal = pitch_shift(signal, self.target_sample_rate, n_steps=pitch_steps)

        noise = torch.randn_like(signal, device=self.device)
        snr = torch.randint(20, 30, (1,)).to(self.device)

        signal = add_noise(signal, noise=noise, snr=snr)

        return signal

    def _augment_spectrogram_if_necessary(self, mel_spectrogram):
        if not self.augment:
            return mel_spectrogram

        freq_mask = transforms.FrequencyMasking(freq_mask_param=30).to(self.device)
        time_mask = transforms.TimeMasking(time_mask_param=100).to(self.device)

        mel_spectrogram = freq_mask(mel_spectrogram)
        mel_spectrogram = time_mask(mel_spectrogram)
        return mel_spectrogram

    @staticmethod
    def _mix_down_if_necessary(signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


def get_dataloader(metadata, data_dir, sample_rate=16000, n_mels=128,
                   batch_size=16, num_workers=0, augment=False, audio_length=6, device="cpu"):
    dataset = MozillaCommonVoiceDatasetCNN(metadata, data_dir, sample_rate, n_mels, augment=augment,
                                           audio_length=audio_length, device=device)

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


def get_num_classes(file_path):
    df = pd.read_csv(file_path)
    return len(df["accents"].unique())


def get_classes(metadata_path):
    df = pd.read_csv(metadata_path)
    # get one unique value for each class
    dic = dict(zip(df['accent'], df['accents']))
    # return list ordered by the class number
    return [dic[i] for i in range(len(dic))]


if __name__ == "__main__":
    main()
