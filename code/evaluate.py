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
import matplotlib.pyplot as plt
import h5py
import HighLevelFeatures as HLF

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
plt.rc('font', family='serif')


########## Parser Setup ##########

parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                              'Fast Calorimeter Challenge 2022.'))

parser.add_argument('--input_file', '-i', help='Name of the input file to be evaluated.')
parser.add_argument('--reference_file', '-r',
                    help='Name of the file to be used as reference. Must be .hdf5 when'+\
                    ' run for the first time, can be the .pkl that is created in the first'+\
                    ' run for faster runtime in subsequent runs.')
parser.add_argument('--mode', '-m', default='all',
                    choices=['all', 'avg', 'avg-E', 'hist-p', 'hist-chi', 'hist',
                             'cls-low', 'cls-high'],
                    help=("What metric to evaluate: " +\
                          "'avg' plots the shower average;" +\
                          "'avg-E' plots the shower average for energy ranges;" +\
                          "'hist-p' plots the histograms;" +\
                          "'hist-chi' evaluates a chi2 of the histograms;" +\
                          "'hist' evaluates a chi2 of the histograms and plots them;" +\
                          "'cls-low' trains a classifier on the low-level feautures;" +\
                          "'cls-high' trains a classifier on the high-level features;" +\
                          "'all' does the full evaluation, ie all of the above."))
parser.add_argument('--dataset', '-d', choices=['1-photons', '1-pions', '2', '3'],
                    help='Which dataset is evaluated.')
parser.add_argument('--output_dir', default='evaluation_results/',
                    help='Where to store evaluation output files (plots and scores).')
parser.add_argument('--source_dir', default='source/',
                    help='Folder that contains (soft links to) files required for'+\
                    ' comparative evaluations (high level features stored in .pkl or '+\
                    'datasets prepared for classifier runs.).')


########## Functions and Classes ##########

def check_file(given_file, arg, which=None):
    """ checks if the provided file has the expected structure based on the dataset """
    print("Checking if {} file has the correct form ...".format(
        which if which is not None else 'provided'))
    num_features = {'1-photons': 368, '1-pions': 533, '2': 6480, '3': 40500}[arg.dataset]
    num_events = given_file['incident_energies'].shape[0]
    assert given_file['showers'].shape[0] == num_events, \
        ("Number of energies provided does not match number of showers, {} != {}".format(
            num_events, given_file['showers'].shape[0]))
    assert given_file['showers'].shape[1] == num_features, \
        ("Showers have wrong shape, expected {}, got {}".format(
            num_features, given_file['showers'].shape[1]))

    print("Found {} events in the file.".format(num_events))
    print("Checking if {} file has the correct form: DONE \n".format(
        which if which is not None else 'provided'))

#def check_reference(arg):
#    """ checks if reference file for comparisons exist """
#    return os.path.exists(os.path.join(arg.source_dir, 'reference_{}.hdf5'.format(arg.dataset)))

#def check_pickle(arg):
#    """ checks if reference pickle file of high-level features exist """
#    return os.path.exists(os.path.join(arg.source_dir, 'reference_{}.pkl'.format(arg.dataset)))

def extract_shower_and_energy(given_file, arg, which):
    """ reads .hdf5 file and returns samples and their energy """
    print("Extracting showers from {} file ...".format(which))
    shower = given_file['showers'][:]
    energy = given_file['incident_energies'][:]
    print("Extracting showers from {} file: DONE.\n".format(which))
    return shower, energy

#def create_reference(arg):
#    """ Create pickle file with high-level features for reference in plots """
#    print("Could not find pickle file with high-level features as reference. Creating it now ...")
#    particle = {'1-photons': 'photon', '1-pions': 'pion',
#                '2': 'electron', '3': 'electron'}[arg.dataset]
#    hlf_ref = HLF.HighLevelFeatures(particle,
#                                    filename='binning_dataset_{}.xml'.format(
#                                        arg.dataset.replace('-', '_')))
#    source_file = h5py.File(os.path.join(arg.source_dir, 'reference_{}.hdf5'.format(arg.dataset)),
#                            'r')
#    reference_showers, reference_energies = extract_shower_and_energy(source_file, arg)
#    hlf_ref.CalculateFeatures(reference_showers)
#    hlf_ref.Einc = reference_energies
#    with open(os.path.join(arg.source_dir, 'reference_{}.pkl'.format(arg.dataset)), 'wb') as file:
#        pickle.dump(hlf_ref, file)
#    print("Creating reference pickle file: DONE.\n")
#    return hlf_ref

def load_reference(filename):
    """ Load existing pickle with high-level features for reference in plots """
    print("Loading file with high-level features.")
    with open(filename, 'rb') as file:
        hlf_ref = pickle.load(file)
    return hlf_ref

def save_reference(ref_hlf, ref_name, arg):
    """ Saves high-level features class to file """
    print("Saving file with high-level features.")
    filename = os.path.splitext(os.path.basename(ref_name))[0] + '.pkl'
    with open(os.path.join(arg.source_dir, filename), 'wb') as file:
        pickle.dump(ref_hlf, file)
    print("Saving file with high-level features DONE.")

def plot_histograms(hlf_class, reference_class, arg):
    """ plots histograms based with reference file as comparison """
    plot_Etot_Einc(hlf_class, reference_class, arg)
    plot_E_layers(hlf_class, reference_class, arg)
    plot_ECEtas(hlf_class, reference_class, arg)
    plot_ECPhis(hlf_class, reference_class, arg)
    plot_ECWidthEtas(hlf_class, reference_class, arg)
    plot_ECWidthPhis(hlf_class, reference_class, arg)

def plot_Etot_Einc(hlf_class, reference_class, arg):
    """ plots Etot normalized to Einc histogram """

    bins = np.linspace(0.5, 1.5, 101)
    plt.figure(figsize=(6, 6))
    counts_ref, _, _ = plt.hist(reference_class.GetEtot() / reference_class.Einc.squeeze(),
                                bins=bins, label='reference', density=True,
                                histtype='stepfilled', alpha=0.2, linewidth=2.)
    counts_data, _, _ = plt.hist(hlf_class.GetEtot() / hlf_class.Einc.squeeze(), bins=bins,
                                 label='generated', histtype='step', linewidth=3., alpha=1.,
                                 density=True)
    plt.xlim(0.5, 1.5)
    plt.xlabel(r'$E_{\text{tot}} / E_{\text{inc}}$')
    plt.legend(fontsize=20)
    plt.tight_layout()
    if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}.png'.format(arg.dataset))
        plt.savefig(filename, dpi=300)
    if arg.mode in ['all', 'hist-chi', 'hist']:
        seps = separation_power(counts_ref, counts_data, bins)
        print("Separation power of Etot / Einc histogram: {}".format(seps))
        with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                  'a') as f:
            f.write('Etot / Einc: \n')
            f.write(str(seps))
            f.write('\n\n')
    plt.close()


def plot_E_layers(hlf_class, reference_class, arg):
    """ plots energy deposited in each layer """
    for key in hlf_class.GetElayers().keys():
        plt.figure(figsize=(6, 6))
        counts_ref, bins, _ = plt.hist(reference_class.GetElayers()[key], bins=20,
                                       label='reference', density=True, histtype='stepfilled',
                                       alpha=0.2, linewidth=2.)
        counts_data, _, _ = plt.hist(hlf_class.GetElayers()[key], label='generated', bins=bins,
                                     histtype='step', linewidth=3., alpha=1., density=True)
        plt.title("Energy deposited in layer {}".format(key))
        plt.xlabel(r'$E$ [MeV]')
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir, 'E_layer_{}_dataset_{}.png'.format(
                key,
                arg.dataset))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = separation_power(counts_ref, counts_data, bins)
            print("Separation power of E layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('E layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECEtas(hlf_class, reference_class, arg):
    """ plots center of energy in eta """
    for key in hlf_class.GetECEtas().keys():
        if arg.dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-500., 500.)
        else:
            lim = (-100., 100.)
        plt.figure(figsize=(6, 6))
        bins = np.linspace(*lim, 101)
        counts_ref, _, _ = plt.hist(reference_class.GetECEtas()[key], bins=bins,
                                    label='reference', density=True, histtype='stepfilled',
                                    alpha=0.2, linewidth=2.)
        counts_data, _, _ = plt.hist(hlf_class.GetECEtas()[key], label='generated', bins=bins,
                                     histtype='step', linewidth=3., alpha=1., density=True)
        plt.title(r"Center of Energy in $\Delta\eta$ in layer {}".format(key))
        plt.xlabel(r'[mm]')
        plt.xlim(*lim)
        plt.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'ECEta_layer_{}_dataset_{}.png'.format(key,
                                                                           arg.dataset))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = separation_power(counts_ref, counts_data, bins)
            print("Separation power of EC Eta layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('EC Eta layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECPhis(hlf_class, reference_class, arg):
    """ plots center of energy in phi """
    for key in hlf_class.GetECPhis().keys():
        if arg.dataset in ['2', '3']:
            lim = (-30., 30.)
        elif key in [12, 13]:
            lim = (-500., 500.)
        else:
            lim = (-100., 100.)
        plt.figure(figsize=(6, 6))
        bins = np.linspace(*lim, 101)
        counts_ref, _, _ = plt.hist(reference_class.GetECPhis()[key], bins=bins,
                                    label='reference', density=True, histtype='stepfilled',
                                    alpha=0.2, linewidth=2.)
        counts_data, _, _ = plt.hist(hlf_class.GetECPhis()[key], label='generated', bins=bins,
                                     histtype='step', linewidth=3., alpha=1., density=True)
        plt.title(r"Center of Energy in $\Delta\phi$ in layer {}".format(key))
        plt.xlabel(r'[mm]')
        plt.xlim(*lim)
        plt.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'ECPhi_layer_{}_dataset_{}.png'.format(key,
                                                                           arg.dataset))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = separation_power(counts_ref, counts_data, bins)
            print("Separation power of EC Phi layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('EC Phi layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECWidthEtas(hlf_class, reference_class, arg):
    """ plots width of center of energy in eta """
    for key in hlf_class.GetWidthEtas().keys():
        if arg.dataset in ['2', '3']:
            lim = (0., 30.)
        elif key in [12, 13]:
            lim = (0., 400.)
        else:
            lim = (0., 100.)
        plt.figure(figsize=(6, 6))
        bins = np.linspace(*lim, 101)
        counts_ref, _, _ = plt.hist(reference_class.GetWidthEtas()[key], bins=bins,
                                    label='reference', density=True, histtype='stepfilled',
                                    alpha=0.2, linewidth=2.)
        counts_data, _, _ = plt.hist(hlf_class.GetWidthEtas()[key], label='generated', bins=bins,
                                     histtype='step', linewidth=3., alpha=1., density=True)
        plt.title(r"Width of Center of Energy in $\Delta\eta$ in layer {}".format(key))
        plt.xlabel(r'[mm]')
        plt.xlim(*lim)
        plt.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'WidthEta_layer_{}_dataset_{}.png'.format(key,
                                                                              arg.dataset))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = separation_power(counts_ref, counts_data, bins)
            print("Separation power of Width Eta layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('Width Eta layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

def plot_ECWidthPhis(hlf_class, reference_class, arg):
    """ plots width of center of energy in phi """
    for key in hlf_class.GetWidthPhis().keys():
        if arg.dataset in ['2', '3']:
            lim = (0., 30.)
        elif key in [12, 13]:
            lim = (0., 400.)
        else:
            lim = (0., 100.)
        plt.figure(figsize=(6, 6))
        bins = np.linspace(*lim, 101)
        counts_ref, _, _ = plt.hist(reference_class.GetWidthPhis()[key], bins=bins,
                                    label='reference', density=True, histtype='stepfilled',
                                    alpha=0.2, linewidth=2.)
        counts_data, _, _ = plt.hist(hlf_class.GetWidthPhis()[key], label='generated', bins=bins,
                                     histtype='step', linewidth=3., alpha=1., density=True)
        plt.title(r"Width of Center of Energy in $\Delta\phi$ in layer {}".format(key))
        plt.xlabel(r'[mm]')
        plt.xlim(*lim)
        plt.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'WidthPhi_layer_{}_dataset_{}.png'.format(key,
                                                                              arg.dataset))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = separation_power(counts_ref, counts_data, bins)
            print("Separation power of Width Phi layer {} histogram: {}".format(key, seps))
            with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                      'a') as f:
                f.write('Width Phi layer {}: \n'.format(key))
                f.write(str(seps))
                f.write('\n\n')
        plt.close()

#def download_source_reference():
#    """ Download dataset needed for reference """
#    raise NotImplementedError()

def separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()

########## Main ##########

if __name__ == '__main__':
    args = parser.parse_args()
    #print(vars(args))

    source_file = h5py.File(args.input_file, 'r')
    check_file(source_file, args, which='input')

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.isdir(args.source_dir):
        os.makedirs(args.source_dir)

    particle = {'1-photons': 'photon', '1-pions': 'pion',
                '2': 'electron', '3': 'electron'}[args.dataset]
    hlf = HLF.HighLevelFeatures(particle,
                                filename='binning_dataset_{}.xml'.format(
                                    args.dataset.replace('-', '_')))

    if os.path.splitext(args.reference_file)[1] == '.hdf5':
        print("using .hdf5 reference")
        reference_file = h5py.File(args.reference_file, 'r')
        check_file(reference_file, args, which='reference')
        reference_hlf = HLF.HighLevelFeatures(particle,
                                              filename='binning_dataset_{}.xml'.format(
                                                  args.dataset.replace('-', '_')))
        reference_shower, reference_energy = extract_shower_and_energy(reference_file, args,
                                                                       which='reference')
        reference_hlf.Einc = reference_energy
        save_reference(reference_hlf, args.reference_file, args)

    elif os.path.splitext(args.reference_file)[1] == '.pkl':
        print("using .pkl file for reference")
        reference_hlf = load_reference(args.reference_file)
    else:
        raise ValueError("reference_file must be .hdf5 or .pkl!")

    shower, energy = extract_shower_and_energy(source_file, args, which='input')

    # evaluations:
    if args.mode in ['all', 'avg']:
        print("Plotting average shower...")
        _ = hlf.DrawAverageShower(shower,
                                  filename=os.path.join(args.output_dir,
                                                        'average_shower_dataset_{}.png'.format(
                                                            args.dataset)),
                                  title="Shower average")
        if hasattr(reference_hlf, 'avg_shower'):
            pass
        else:
            reference_hlf.avg_shower = reference_shower.mean(axis=0, keepdims=True)
            save_reference(reference_hlf, args.reference_file, args)
        _ = hlf.DrawAverageShower(reference_hlf.avg_shower,
                                  filename=os.path.join(
                                      args.output_dir,
                                      'reference_average_shower_dataset_{}.png'.format(
                                          args.dataset)),
                                  title="Shower average reference dataset")
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
            if hasattr(reference_hlf, 'avg_shower_E'):
                pass
            else:
                reference_hlf.avg_shower_E = {}
            if target_energies[i] in reference_hlf.avg_shower_E:
                pass
            else:
                which_showers = ((reference_hlf.Einc >= target_energies[i]) & \
                             (reference_hlf.Einc < target_energies[i+1])).squeeze()
                reference_hlf.avg_shower_E[target_energies[i]] = \
                    reference_shower[which_showers].mean(axis=0, keepdims=True)
                save_reference(reference_hlf, args.reference_file, args)

            _ = hlf.DrawAverageShower(reference_hlf.avg_shower_E[target_energies[i]],
                                      filename=os.path.join(args.output_dir,
                                                            'reference_'+filename),
                                      title='reference '+plot_title[i])

        print("Plotting average shower for different energies: DONE.\n")

    if args.mode in ['all', 'hist-p', 'hist-chi', 'hist']:
        print("Calculating high-level features for histograms ...")
        hlf.CalculateFeatures(shower)
        hlf.Einc = energy

        if reference_hlf.E_tot is None:
            reference_hlf.CalculateFeatures(reference_shower)
            save_reference(reference_hlf, args.reference_file, args)
        print("Calculating high-level features for histograms: DONE.\n")

        if args.mode in ['all', 'hist-chi', 'hist']:
            with open(os.path.join(args.output_dir, 'histogram_chi2_{}.txt'.format(args.dataset)),
                      'w') as f:
                f.write('List of chi2 of the plotted histograms,'+\
                        ' see eq. 15 of 2009.03796 for its definition.\n')
        print("Plotting histograms ...")
        plot_histograms(hlf, reference_hlf, args)
        print("Plotting histograms: DONE. \n")

    if args.mode in ['cls-low', 'cls-high']:
        raise NotImplementedError("Stay Tuned!")
