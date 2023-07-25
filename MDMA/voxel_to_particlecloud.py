import argparse
import os
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler


def build_particle_clouds(tensor, n):
    """
    Converts hits with shower ids to particle clouds.
    """
    cumsum = torch.cat([torch.tensor([0]), n.cumsum(dim=0)])
    point_clouds = [tensor[cumsum[i] : cumsum[i + 1]] for i in range(len(cumsum) - 1)]
    return point_clouds


class DQ(BaseEstimator, TransformerMixin):
    """
    Dequantises the data by adding a random number between 0 and 1 to each coordinate of a hit.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        X = X + np.random.rand(*X.shape)
        return X

    def inverse_transform(self, X, y=None):
        X = np.floor(X)
        return X


class ScalerBaseNew:
    """
    Base Scaler class to transform and save data
    """

    def __init__(self, transformers, name, overwrite=False, data_dir="./"):
        """
        Initialize the ScalerBaseNews.
        """
        self.transformers = transformers
        self.data_dir = data_dir
        self.scaler_path = Path(data_dir) / f"scaler_{name}.gz"
        self.name = name
        self.n_features = 4
        # Load existing transformers if they exist and overwrite flag is False
        if self.scaler_path.is_file() and not overwrite:
            self.transformers = joblib.load(self.scaler_path)

    def save_scalar(self, point_cloud):
        """
        Fits transformers, transforms the data and saves the scaler.
        """
        device = point_cloud.device
        original_shape = point_cloud.shape
        point_cloud = point_cloud.cpu().numpy().astype(np.float64).reshape(-1, self.n_features)
        point_cloud = np.hstack([self.transformers[0].fit_transform(point_cloud[:, :1]), self.transformers[1].fit_transform(point_cloud[:, 1:])])
        # Save transformers
        joblib.dump(self.transformers, self.scaler_path)
        return torch.from_numpy(point_cloud).reshape(*original_shape).to(device).float()

    def transform(self, point_cloud):
        """
        Transform the data with the saved transformers, takes a torch tensor as input and outputs a tensor
        """
        original_shape = point_cloud.shape
        device = point_cloud.device
        point_cloud = point_cloud.cpu().numpy().astype(np.float64).reshape(-1, self.n_features)
        return torch.from_numpy(np.hstack([self.transformers[0].transform(point_cloud[:, :1]), self.transformers[1].transform(point_cloud[:, 1:])]).reshape(*original_shape)).to(device).float()


class LogitTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms data using logit and inverse logit transformations.
    """

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.log(X / (1 - X))

    def inverse_transform(self, X, y=None):
        return 1 / (1 + np.exp(-X))

    def check_inverse(self, X):
        assert np.allclose(self.transform(self.inverse_transform(X)), X)


def parse_args():
    parser = argparse.ArgumentParser(description="Transform a voxel representation to a point cloud and dequantize it.")
    parser.add_argument("--dataset_name", required=True, type=str, help="Name of the dataset to process")
    parser.add_argument("--files", required=True, type=str, nargs="+", help="List of input files to process")
    parser.add_argument("--in_dir", required=True, type=str, help="Path to the input directory")
    parser.add_argument("--out_dir", required=True, type=str, help="Path to the output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset_name
    files = args.files
    in_dir = args.in_dir
    out_dir = args.out_dir
    num_z = 45
    num_alpha = 16 if "dataset_2" in dataset_name else 50
    num_r = 9 if "dataset_2" in dataset_name else 18

    with torch.no_grad():
        for mode in ["train", "test"]:
            if mode == "train":
                scaler = ScalerBaseNew(transformers=[PowerTransformer(method="box-cox", standardize=True), Pipeline([("minmax", MinMaxScaler(feature_range=(1e-5, 1 - 1e-5))), ("logit_transformer", LogitTransformer()), ("std", StandardScaler())])], name=dataset_name, overwrite=True, data_dir=out_dir)

            energies, showers = [], []
            for file in files:
                with h5py.File(in_dir + "/" + file, "r") as electron_file:
                    energies.append(electron_file["incident_energies"][:])
                    showers.append(electron_file["showers"][:])

            energies = np.concatenate(energies) if len(energies) > 1 else energies[0]
            showers = np.concatenate(showers).reshape(-1, num_z, num_alpha, num_r) if len(showers) > 1 else showers[0].reshape(-1, num_z, num_alpha, num_r)

            # Filter out zeros from showers
            showers_mask = (showers > 0).any(1).any(1).any(1)
            energies = energies[showers_mask]
            showers = showers[showers_mask]

            # Extract non-zero coordinates and values
            coords = np.argwhere(showers > 0.0)  # get indices of non-zero values (shower_id, r, alpha, z)
            vals = showers[coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]]  # get non-zero values
            _, nnz = np.unique(coords[:, 0], return_counts=True)  # get number of hits per shower
            nnz = torch.from_numpy(nnz)

            # Remove shower_id from coords
            coords = coords[:, 1:]

            # Calculate start_index
            start_index = np.zeros(nnz.shape, dtype=np.int64)
            start_index[1:] = np.cumsum(nnz)[:-1]

            data = torch.from_numpy(np.hstack((vals[:, None], coords)))

            # Fit and transform data if in train mode
            if mode == "train":
                data = scaler.save_scalar(data)
            else:
                data = scaler.transform(data)
            particle_cloud = build_particle_clouds(data, nnz)

            # Save the particle clouds, incoming energies and n

            torch.save({"data": particle_cloud, "energies": torch.from_numpy(energies), "n": nnz}, f"{out_dir}/pc_{mode}_{dataset_name}.pt", pickle_protocol=4)
