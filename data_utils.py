import os
import random
import numpy as np
import torch
import torch.utils.data
import phkit
import utils

import layers
from utils import load_wav_to_torch, load_filepaths_and_text, load_text
from text import text_to_sequence

from encoder.audio import preprocess_wav


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(
                audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class ASDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, hparams, mode):
        self.dataset_path = dataset_path
        self.set_mode(mode)

        self.data_list = utils.load_text(self.dataset_path + "/content.txt")

        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax
        )

        random.seed(hparams.seed)
        random.shuffle(self.data_list)

        split_ind = int(len(self.data_list)*hparams.train_set_ratio)
        if mode == "train":
            self.data_list = self.data_list[:split_ind]
        elif mode == "val":
            self.data_list = self.data_list[split_ind:]

        utils.print_INFO(
            "AS_dataset", "load {} data list successfully".format(self.mode)
        )

    def set_mode(self, mode):
        assert mode in ("train", "val", "test")
        self.mode = mode
        if mode == "train" or mode == "val":
            self.dataset_path += "/train"
        elif mode == "test":
            self.dataset_path += "/test"

    def get_mel(self, path):
        # wav, sampling_rate = load_wav_to_torch(path)
        # if sampling_rate != self.stft.sampling_rate:
        #     raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, self.stft.sampling_rate))
        # wav = wav / self.max_wav_value
        wav, _ = utils.load_wav(path, sr=self.sampling_rate)
        wav = torch.FloatTensor(wav)
        wav = wav.unsqueeze(0)
        wav = torch.autograd.Variable(wav, requires_grad=False)
        mel = self.stft.mel_spectrogram(wav)
        mel = mel.squeeze(0)
        return mel

    def get_text(self, text):
        text = text.split(" ")
        text = "".join(text[0: len(text): 2])
        sequence = phkit.text2sequence(text)
        return torch.IntTensor(sequence)

    def __getitem__(self, index):
        filename, text = self.data_list[index].split("\t")
        speaker = filename[:7]
        text = text[:-1]
        speaker_path = self.dataset_path + "/wav/{}/".format(speaker)
        speaker_wav_files = os.listdir(speaker_path)
        refer_wav_path = random.choice(speaker_wav_files)
        path = speaker_path + filename
        refer_wav_path = speaker_path + refer_wav_path

        refer_wav = preprocess_wav(refer_wav_path, source_sr=self.sampling_rate)

        return (self.get_text(text), self.get_mel(path), refer_wav)

    def __len__(self):
        return len(self.data_list)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        # FIXME: preprocess the embedding wav path
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - \
                max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths
