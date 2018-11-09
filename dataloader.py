import numpy as np
import torch
from skimage.transform import rescale
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tqdm import tqdm


def load_sample(url):
    x = np.load(url)
    x = rescale(x, 0.55, multichannel=False, anti_aliasing=True, mode='reflect')
    x = torch.from_numpy(x)
    return x


def normalize_sample(x):
    """
    Normalization parameters found using compute_dataset_mean_std()
    :param x: spectrogram
    :return: whitened spectrogram
    """
    x = (x - 4.6242e-48) / 2.0073e-46
    return x


def compute_dataset_mean_std():
    load = lambda x: torch.from_numpy(np.load(x))
    dataset = DatasetFolder('data/train', load, ('.npy',))
    loader = DataLoader(dataset, batch_size=64, num_workers=8)

    sum = torch.zeros_like(dataset[0][0]).to('cuda')
    ntotal = 0

    progress = tqdm(loader)
    progress.set_description('Computing MEAN')
    for x, y in progress:
        x = x.to('cuda')
        sum += x.sum(0)
        ntotal += y.shape[0]

    mean = (sum / ntotal).mean()
    print(mean)

    sum = 0
    progress = tqdm(loader)
    progress.set_description('Computing STD')
    for x, y in progress:
        x = x.to('cuda')
        sum += ((x - mean)**2).sum(0)

    std = torch.sqrt((sum / ntotal).mean())
    print(std)


def compute_data_stats():
    dataset = DatasetFolder('data/train', load_sample, ('.npy',), transform=normalize_sample())
    for i in dataset:
        print(i.min(), i.max(), i.mean(), i.std())


if __name__ == '__main__':
    compute_dataset_mean_std()
    # compute_data_stats()