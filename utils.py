import numpy as np
import librosa
from scipy.io.wavfile import read
import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def load_wav(path, sr):
    return librosa.load(path, sr=sr)


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.readlines()
    return text


def print_ERROR(location, message):
    print("\033[1;31m[ERROR]\033[0m\033[1;36m[{}]\033[0m: {}".format(
        location, message))


def print_WARNING(location, message):
    print("\033[1;33m[WARNING]\033[0m\033[1;36m[{}]\033[0m: {}".format(
        location, message))


def print_INFO(location, message):
    print("\033[1;34m[INFO]\033[0m\033[1;36m[{}]\033[0m: {}".format(
        location, message))
