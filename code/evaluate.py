# pylint: disable=invalid-name
""" Main script to evaluate contributions to the Fast Calorimeter Challenge 2022

    input:
        - set of events in .hdf5 file format
    output:
        - metrics for evaluation (plots, classifier scores, etc.)

"""

import argparse
import os
import pickle

import numpy as np
import h5py
import HighLevelFeatures as HLF



########## Parser Setup ##########

parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                              'Fast Calorimeter Challenge 2022.'))

parser.add_argument('--input_file', '-i', help='Name of the input file to be evaluated.')
parser.add_argument('--mode', '-m', default='all',
                    choices=['all', 'avg', 'avg-E', 'hist-p', 'hist-chi', 'cls-low', 'cls-high'],
                    help=("What metric to evaluate: " +\
                          "'avg' plots the shower average;" +\
                          "'avg-E' plots the shower average for energy ranges;" +\
                          "'hist-p' plots the histograms;" +\
                          "'hist-chi' evaluates a chi2 of the histograms;" +\
                          "'cls-low' trains a classifier on the low-level feautures;" +\
                          "'cls-high' trains a classifier on the high-level features;" +\
                          "'all' does the full evaluation, ie all of the above."))
parser.add_argument('--dataset', '-d', choices=['1-photons', '1-pions', '2', '3'],
                    help='Which dataset is evaluated.')
parser.add_argument('--output_dir', default='evaluation_results/',
                    help='Where to store evaluation output files (plots and scores).')
parser.add_argument('--source_dir', default='source/',
                    help='Folder that contains files required for comparative evaluations.')


########## Functions and Classes ##########

def check_file(given_file, arg):
    """ checks if the provided file has the expected structure based on the dataset """
    print("Checking if provided file has the correct form ...")
    num_features = {'1-photons': 368, '1-pions': 533, '2': 6480, '3': 40500}[arg.dataset]
    num_events = given_file['incident_energies'].shape[0]
    assert given_file['showers'].shape[0] == num_events, \
        ("Number of energies provided does not match number of showers, {} != {}".format(
            num_events, given_file['showers'].shape[0]))
    assert given_file['showers'].shape[1] == num_features, \
        ("Showers have wrong shape, expected {}, got {}".format(
            num_features, given_file['showers'].shape[1]))

    print("Found {} events in the file.".format(num_events))
    print("Checking if provided file has the correct form: DONE \n")

def check_reference(arg):
    """ checks if reference file for comparisons exist """
    return os.path.exists(os.path.join(args.source_dir, 'reference_{}.hdf5'.format(arg.dataset)))

def check_pickle(arg):
    """ checks if reference pickle file of high-level features exist """
    return os.path.exists(os.path.join(args.source_dir, 'reference_{}.pkl'.format(arg.dataset)))

def extract_shower_and_energy(given_file, arg):
    """ reads .hdf5 file and returns samples and their energy """
    print("Extracting showers from file...")
    shower = given_file['showers'][:]
    energy = given_file['incident_energies'][:]
    print("Extracting showers from file: DONE.\n")
    return shower, energy

def create_reference(arg):
    """ Create pickle file with high-level features for reference in plots """
    print("Could not find pickle file with high-level features as reference. Creating it now ...")
    particle = {'1-photons': 'photon', '1-pions': 'pion',
                '2': 'electron', '3': 'electron'}[arg.dataset]
    hlf_ref = HLF.HighLevelFeatures(particle,
                                    filename='binning_dataset_{}.xml'.format(
                                        arg.dataset.replace('-', '_')))
    source_file = h5py.File(os.path.join(arg.source_dir, 'reference_{}.hdf5'.format(arg.dataset)),
                            'r')
    reference_showers, reference_energies = extract_shower_and_energy(source_file, arg)
    hlf_ref.CalculateFeatures(reference_showers)
    with open(os.path.join(arg.source_dir, 'reference_{}.pkl'.format(arg.dataset)), 'wb') as file:
        pickle.dump(hlf_ref, file)
    print("Creating reference pickle file: DONE")
    return hlf_ref

def load_reference(arg):
    """ Load existing pickle with high-level features for reference in plots """
    print("Loading file with high-level features.")
    with open(os.path.join(arg.source_dir, 'reference_{}.pkl'.format(arg.dataset)), 'rb') as file:
        hlf_ref = pickle.load(file)
    return hlf_ref

def plot_histograms(hlf_class, reference_class, arg):
    """ plots histograms based with reference file as comparison """
    plot_Etot_Einc(hlf_class, reference_class, arg)

def plot_Etot_Einc(hlf_class, reference_class, arg):
    """ plots Etot normalized to Einc histogram """
    pass

def download_source_reference():
    raise NotImplementedError()

########## Main ##########

if __name__ == '__main__':
    args = parser.parse_args()
    #print(vars(args))

    source_file = h5py.File(args.input_file, 'r')

    check_file(source_file, args)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    shower, energy = extract_shower_and_energy(source_file, args)
    particle = {'1-photons': 'photon', '1-pions': 'pion',
                '2': 'electron', '3': 'electron'}[args.dataset]
    hlf = HLF.HighLevelFeatures(particle,
                                filename='binning_dataset_{}.xml'.format(
                                    args.dataset.replace('-', '_')))

    # evaluations:
    if args.mode in ['all', 'avg']:
        print("Plotting average shower...")
        _ = hlf.DrawAverageShower(shower,
                                  filename=os.path.join(args.output_dir,
                                                        'average_shower_dataset_{}.png'.format(
                                                            args.dataset)),
                                  title="Shower average")
        print("Plotting average shower: DONE.\n")

    if args.mode in ['all', 'avg-E']:
        print("Plotting average showers for different energies ...")
        if '1' in args.dataset:
            target_energies = 2**np.linspace(8, 23, 16)
            plot_title = ['shower average at E = {} MeV'.format(int(en)) for en in target_energies]
        else:
            target_energies = 10**np.linspace(3, 6, 4)
            plot_title = []
            for i in range(3, 7):
                plot_title.append('shower average for E in [{}, {}] MeV'.format(10**i, 10**(i+1)))
        for i in range(len(target_energies)-1):
            filename = 'average_shower_dataset_{}_E_{}.png'.format(args.dataset,
                                                                   target_energies[i])
            which_showers = ((energy >= target_energies[i]) & \
                             (energy < target_energies[i+1])).squeeze()
            _ = hlf.DrawAverageShower(shower[which_showers],
                                      filename=os.path.join(args.output_dir, filename),
                                      title=plot_title[i])
        print("Plotting average shower for different energies: DONE.\n")

    # any further evaluation needs a reference dataset, check if that exists:
    if check_reference(args):
        print("Reference hdf5 file exist in {}, moving on.".format(args.source_dir))
    else:
        # TODO: ask if download is ok and download
        raise FileNotFoundError(
            "Reference hdf5 file does not exist in {}, please provide file {}".format(
                args.source_dir, 'reference_{}.hdf5'.format(args.dataset)))

    if args.mode in ['all', 'hist-p']:
        print("Calculating high-level features for histograms ...")
        hlf.CalculateFeatures(shower)
        print("Calculating high-level features for histograms: DONE")
        if check_pickle(args):
            reference = load_reference(args)
        else:
            reference = create_reference(args)
        print("Plotting histograms...")
        plot_histograms(hlf, reference, args)
