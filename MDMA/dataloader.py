import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset

from voxel_to_particlecloud import DQ, LogitTransformer, ScalerBaseNew


class CustomDataset(Dataset):
    def __init__(self, data, E):
        assert len(data) == len(E), "The lengths of data and E are not equal"
        self.data = data
        self.E = E

    def __getitem__(self, index):
        return self.data[index], self.E[index]

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    """
    This is a custom batch sampler that sorts the sequences by length and
    creates batches based on the sorted indices. This is necessary for
    efficient padding of the sequences.
    """

    def __init__(self, data_source, batch_size, shuffle=True, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        # Sort sequences by length
        indices = sorted(indices, key=lambda x: len(self.data_source[x]))
        # Create batches based on the sorted indices
        batches = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        if self.drop_last or len(batches[-1]) == 0:
            batches = batches[:-1]
        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def pad_collate_fn(batch):
    """this is the collate function that is used by the dataloader to pad the sequences to equal length
    also returns the mask that is used to mask out the padded elements and the incoming energy"""
    batch, E = zip(*batch)
    max_len = max(len(sample) for sample in batch)
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0.0)[:, :, :4].float()
    mask = ~(torch.arange(max_len).expand(len(batch), max_len) < torch.tensor([len(sample) for sample in batch]).unsqueeze(1))
    E = (torch.stack(E).log() - 10).float()
    return padded_batch, mask, E


class ParticleCloudDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
    """

    def __init__(self, name, batch_size, in_dir, **kwargs):
        self.name = name
        self.in_dir = in_dir
        self.batch_size = batch_size
        super().__init__()

    def setup(self, stage):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        self.data = torch.load(f"{self.in_dir}/pc_train_{self.name}.pt")
        self.E = self.data["energies"]
        self.data = self.data["data"]
        self.val_data = torch.load(f"{self.in_dir}/pc_test_{self.name}.pt")
        self.val_E = self.val_data["energies"]
        self.val_data = self.val_data["data"]

        self.scaler = ScalerBaseNew(transformers=[], name=self.name, overwrite=False)

        self.train_iterator = BucketBatchSampler(self.data, batch_size=self.batch_size, drop_last=True, shuffle=True)
        self.val_iterator = BucketBatchSampler(self.val_data, batch_size=self.batch_size * 4, drop_last=False, shuffle=True)
        self.train_dl = DataLoader(CustomDataset(self.data, self.E), batch_sampler=self.train_iterator, collate_fn=pad_collate_fn, num_workers=16)
        self.val_dl = DataLoader(CustomDataset(self.val_data, self.val_E), batch_sampler=self.val_iterator, collate_fn=pad_collate_fn, num_workers=16)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl


if __name__ == "__main__":
    loader = ParticleCloudDataloader("groundtruth_dataset_3", 64, in_dir="/beegfs/desy/user/kaechben/testing/")
    loader.setup("train")
    mins = torch.ones(4).unsqueeze(0)
    n = []
    for i in loader.train_dataloader():
        n.append((~i[1]).sum(1))
        mins = torch.min(torch.cat((mins, i[0][~i[1]].min(0, keepdim=True)[0]), dim=0), dim=0)[0].unsqueeze(0)
        assert (i[0] == i[0]).all()
    print(torch.cat(n).float().mean())
