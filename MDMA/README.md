# Minimal Scripts to use a particle cloud discriminator

`voxel_to_particlecloud.py` takes as input an voxel hdf in the usual format as in the calochallenge and creates a particle cloud dataset out of it where the data is scaled
`discriminator.py` is the particle cloud discriminator used in MDMA [1].
`dataloader.py` is an example particle cloud datalaoder, which buckets together particle clouds of similar size to reduce padding
`test.py` is an example script on how the multiclassification could be done - for reference this takes around 3 epochs to reach an accuracy of 99.99% on a dataset, where the current FC classifier gets 80% accuracy

To convert a voxel hdf to a particle cloud hdf the following command should be used:
```
python voxel_to_particlecloud.py --files vox_filename.hdf5 --in_dir /path/to/vox --out_dir /path/to/out --dataset_name name
```
`--files`: can be multiple files, space separated (as necessary for dataset 3)
`--in_dir`: directory where voxel files lie
`--out_dir`: directory where particle cloud files should be put
` --dataset_name`: name of outputdataset, note that it is important to add dataset_2 or dataset_3 as a pre/postfix for the dataloader to work out of the box e.g. MDMA_dataset_2

[1] https://arxiv.org/pdf/2305.15254.pdf