# pylint: disable=invalid-name
""" Main script to evaluate contributions to the Fast Calorimeter Challenge 2022

    input:
        - set of events in .hdf5 file format
    output:
        - metrics for evaluation (plots, classifier scores, etc.)

"""

import argparse
import os

import numpy as np
import h5py
import HighLevelFeatures as HLF



########## Parser Setup ##########

parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                              'Fast Calorimeter Challenge 2022.'))

parser.add_argument('--input_file', '-i', help='Name of the input file')
parser.add_argument('--mode', '-m', default='all',
                    choices=['all', 'avg', 'hist', 'classifier-low', 'classifier-high'],
                    help='What metric to evaluate.')
parser.add_argument('--dataset', '-d', choices=['1-photon', '1-pion', '2', '3'],
                    help='Which dataset is evaluated.')
parser.add_argument('--output_dir', default='evaluation_results/',
                    help='Where to store evaluation output files.')
parser.add_argument('--source_dir', default='source/',
                    help='Folder that contains files required for comparative evaluations.')


########## Functions and Classes ##########

def check_file(given_file, arg):
    """ checks if the provided file has the expected structure based on the dataset """
    print("Checking if provided file has the correct form ...")
    if arg.dataset in ['2', '3']:
        num_events = check_dataset_23(given_file, arg)
        print("Found {} events in the file.".format(num_events))
    else:
        found_keys = check_dataset_1(given_file, arg)
        print("Found {} energies in the file:".format(len(found_keys)))
        for key in found_keys:
            assert key[:5] == 'data_', "Datasets must start with 'data_'."
            print("Energy {} has {} events".format(key[5:], given_file[key].shape[0]))
    print("Finished checking, looks good so far!\n")

def check_dataset_1(given_file, arg):
    """ checks if the provided file has the expected structure of dataset 1"""
    num_features = {'1-photon': 368, '1-pion': 533}[arg.dataset]
    found_keys = []
    for key in given_file.keys():
        assert given_file[key].shape[1] == num_features, \
            ("{} has wrong shape, expected {}, got {}".format(
                key, num_features, given_file[key].shape[1]))
        found_keys.append(key)
    return found_keys

def check_dataset_23(given_file, arg):
    """ checks if the provided file has the expected structure of dataset 2 or 3"""
    num_features = {'2': 6480, '3': 40500}[arg.dataset]
    num_events = given_file['incident_energies'].shape[0]
    assert given_file['showers'].shape[0] == num_events, \
        ("Number of energies provided does not match number of showers, {} != {}".format(
            num_events, given_file['showers'].shape[0]))
    assert given_file['showers'].shape[1] == num_features, \
        ("Showers have wrong shape, expected {}, got {}".format(
            num_features, given_file['showers'].shape[1]))
    return num_events

def extract_shower_and_energy(given_file, arg):
    """ reads .hdf5 file and returns samples and their energy """
    return None, None

def plot_avg(showers, arg):
    """ Plots the averaged showers """
    pass

# if classifier is selected, check if data source exist, if not, ask if download is ok

########## Main ##########

if __name__ == '__main__':
    args = parser.parse_args()
    print(vars(args))

    source_file = h5py.File(args.input_file, 'r')

    check_file(source_file, args)

    shower, energy = extract_shower_and_energy(source_file, args)
