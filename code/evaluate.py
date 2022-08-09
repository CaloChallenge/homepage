# pylint: disable=invalid-name
""" Main script to evaluate contributions to the Fast Calorimeter Challenge 2022

    input:
        - set of events in .hdf5 file format (same shape as training data)
    output:
        - metrics for evaluation (plots, classifier scores, etc.)

    usage:
        -i --input_file: Name and path of the input file to be evaluated.
        -r --reference_file: Name and path of the reference .hdf5 file. A .pkl file will be
                             created at the same location for faster subsequent evaluations.

"""

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

import HighLevelFeatures as HLF

torch.set_default_dtype(torch.float64)

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')
plt.rc('font', family='serif')


########## Parser Setup ##########

parser = argparse.ArgumentParser(description=('Evaluate calorimeter showers of the '+\
                                              'Fast Calorimeter Challenge 2022.'))

parser.add_argument('--input_file', '-i', help='Name of the input file to be evaluated.')
parser.add_argument('--reference_file', '-r',
                    help='Name and path of the .hdf5 file to be used as reference. '+\
                    'A .pkl file is created at the same location '+\
                    'in the first run for faster runtime in subsequent runs.')
parser.add_argument('--mode', '-m', default='all',
                    choices=['all', 'avg', 'avg-E', 'hist-p', 'hist-chi', 'hist',
                             'cls-low', 'cls-low-normed', 'cls-high'],
                    help=("What metric to evaluate: " +\
                          "'avg' plots the shower average;" +\
                          "'avg-E' plots the shower average for energy ranges;" +\
                          "'hist-p' plots the histograms;" +\
                          "'hist-chi' evaluates a chi2 of the histograms;" +\
                          "'hist' evaluates a chi2 of the histograms and plots them;" +\
                          "'cls-low' trains a classifier on the low-level feautures;" +\
                          "'cls-low-normed' trains a classifier on the low-level feautures" +\
                          " with calorimeter layers normalized to 1;" +\
                          "'cls-high' trains a classifier on the high-level features;" +\
                          "'all' does the full evaluation, ie all of the above" +\
                          " with low-level classifier."))
parser.add_argument('--dataset', '-d', choices=['1-photons', '1-pions', '2', '3'],
                    help='Which dataset is evaluated.')
parser.add_argument('--output_dir', default='evaluation_results/',
                    help='Where to store evaluation output files (plots and scores).')
#parser.add_argument('--source_dir', default='source/',
#                    help='Folder that contains (soft links to) files required for'+\
#                    ' comparative evaluations (high level features stored in .pkl or '+\
#                   'datasets prepared for classifier runs.).')


# classifier options

# not possible since train/test/val split is done differently each time
# to-do: save random-seed to file/read prior to split
#parser.add_argument('--cls_load', action='store_true', default=False,
#                    help='Whether or not load classifier from --output_dir')

#parser.add_argument('--cls_normed', action='store_true',
#                    help='Train classifier on showers normed by layer.')
parser.add_argument('--cls_n_layer', type=int, default=2,
                    help='Number of hidden layers in the classifier, default is 2.')
parser.add_argument('--cls_n_hidden', type=int, default='512',
                    help='Hidden nodes per layer of the classifier, default is 512.')
parser.add_argument('--cls_dropout_probability', type=float, default=0.,
                    help='Dropout probability of the classifier, default is 0.')

parser.add_argument('--cls_batch_size', type=int, default=1000,
                    help='Classifier batch size, default is 1000.')
parser.add_argument('--cls_n_epochs', type=int, default=50,
                    help='Number of epochs to train classifier, default is 50.')
parser.add_argument('--cls_lr', type=float, default=2e-4,
                    help='Learning rate of the classifier, default is 2e-4.')

# CUDA parameters
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')
# todo: check timing of that for dataset2 on pascal2
parser.add_argument('--save_mem', action='store_true',
                    help='Data is moved to GPU batch by batch instead of once in total.')

########## Functions and Classes ##########

class DNN(torch.nn.Module):
    """ NN for vanilla classifier. Does not have sigmoid activation in last layer, should
        be used with torch.nn.BCEWithLogitsLoss()
    """
    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        """ Forward pass through the DNN """
        x = self.layers(x)
        return x

def prepare_low_data_for_classifier(hdf5_file, hlf_class, label, normed=False):
    """ takes hdf5_file, extracts Einc and voxel energies, appends label, returns array """
    if normed:
        E_norm_rep = []
        E_norm = []
        for idx, layer_id in enumerate(hlf_class.GetElayers()):
            E_norm_rep.append(np.repeat(hlf_class.GetElayers()[layer_id].reshape(-1, 1),
                                        hlf_class.num_voxel[idx], axis=1))
            E_norm.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
        E_norm_rep = np.concatenate(E_norm_rep, axis=1)
        E_norm = np.concatenate(E_norm, axis=1)
    voxel, E_inc = extract_shower_and_energy(hdf5_file, label)
    if normed:
        voxel = voxel / (E_norm_rep+1e-16)
        ret = np.concatenate([np.log10(E_inc), voxel, np.log10(E_norm+1e-8),
                              label*np.ones_like(E_inc)], axis=1)
    else:
        voxel = voxel / E_inc
        ret = np.concatenate([np.log10(E_inc), voxel, label*np.ones_like(E_inc)], axis=1)
    return ret

def prepare_high_data_for_classifier(hdf5_file, hlf_class, label):
    """ takes hdf5_file, extracts high-level features, appends label, returns array """
    voxel, E_inc = extract_shower_and_energy(hdf5_file, label)
    E_tot = hlf_class.GetEtot()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate([np.log10(E_inc), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                          Width_etas/1e2, Width_phis/1e2, label*np.ones_like(E_inc)], axis=1)
    return ret

def ttv_split(data1, data2, split=np.array([0.6, 0.2, 0.2])):
    """ splits data1 and data2 in train/test/val according to split,
        returns shuffled and merged arrays
    """
    assert len(data1) == len(data2)
    num_events = (len(data1) * split).astype(int)
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    train1, test1, val1 = np.split(data1, num_events.cumsum()[:-1])
    train2, test2, val2 = np.split(data2, num_events.cumsum()[:-1])
    train = np.concatenate([train1, train2], axis=0)
    test = np.concatenate([test1, test2], axis=0)
    val = np.concatenate([val1, val2], axis=0)
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.random.shuffle(val)
    return train, test, val

def load_classifier(constructed_model, parser_args):
    """ loads a saved model """
    filename = parser_args.mode + '_' + parser_args.dataset + '.pt'
    checkpoint = torch.load(os.path.join(parser_args.output_dir, filename),
                            map_location=parser_args.device)
    constructed_model.load_state_dict(checkpoint['model_state_dict'])
    constructed_model.to(parser_args.device)
    constructed_model.eval()
    print('classifier loaded successfully')
    return constructed_model


def train_and_evaluate_cls(model, data_train, data_test, optim, arg):
    """ train the model and evaluate along the way"""
    best_eval_acc = float('-inf')
    arg.best_epoch = -1
    try:
        for i in range(arg.cls_n_epochs):
            train_cls(model, data_train, optim, i, arg)
            with torch.no_grad():
                eval_acc, _, _ = evaluate_cls(model, data_test)
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                args.best_epoch = i+1
                filename = arg.mode + '_' + arg.dataset + '.pt'
                torch.save({'model_state_dict':model.state_dict()},
                           os.path.join(arg.output_dir, filename))
            if eval_acc == 1.:
                break
    except KeyboardInterrupt:
        # training can be cut short with ctrl+c, for example if overfitting between train/test set
        # is clearly visible
        pass

def train_cls(model, data_train, optim, epoch, arg):
    """ train one step """
    model.train()
    for i, data_batch in enumerate(data_train):
        if args.save_mem:
            data_batch = data_batch[0].to(args.device)
        else:
            data_batch = data_batch[0]
        #input_vector, target_vector = torch.split(data_batch, [data_batch.size()[1]-1, 1], dim=1)
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output_vector, target_vector.unsqueeze(1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % (len(data_train)//2) == 0:
            print('Epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, arg.cls_n_epochs, i, len(data_train), loss.item()))
        # PREDICTIONS
        pred = torch.round(torch.sigmoid(output_vector.detach()))
        target = torch.round(target_vector.detach())
        if i == 0:
            res_true = target
            res_pred = pred
        else:
            res_true = torch.cat((res_true, target), 0)
            res_pred = torch.cat((res_pred, pred), 0)

    print("Accuracy on training set is",
          accuracy_score(res_true.cpu(), res_pred.cpu()))

def evaluate_cls(model, data_test, final_eval=False, calibration_data=None):
    """ evaluate on test set """
    model.eval()
    for j, data_batch in enumerate(data_test):
        if args.save_mem:
            data_batch = data_batch[0].to(args.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = output_vector.reshape(-1)
        target = target_vector.double()
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    BCE = torch.nn.BCEWithLogitsLoss()(result_pred, result_true)
    result_pred = torch.sigmoid(result_pred).cpu().numpy()
    result_true = result_true.cpu().numpy()
    eval_acc = accuracy_score(result_true, np.round(result_pred))
    print("Accuracy on test set is", eval_acc)
    eval_auc = roc_auc_score(result_true, result_pred)
    print("AUC on test set is", eval_auc)
    JSD = - BCE + np.log(2.)
    print("BCE loss of test set is {:.4f}, JSD of the two dists is {:.4f}".format(BCE,
                                                                                  JSD/np.log(2.)))
    if final_eval:
        prob_true, prob_pred = calibration_curve(result_true, result_pred, n_bins=10)
        print("unrescaled calibration curve:", prob_true, prob_pred)
        calibrator = calibrate_classifier(model, calibration_data)
        rescaled_pred = calibrator.predict(result_pred)
        eval_acc = accuracy_score(result_true, np.round(rescaled_pred))
        print("Rescaled accuracy is", eval_acc)
        eval_auc = roc_auc_score(result_true, rescaled_pred)
        print("rescaled AUC of dataset is", eval_auc)
        prob_true, prob_pred = calibration_curve(result_true, rescaled_pred, n_bins=10)
        print("rescaled calibration curve:", prob_true, prob_pred)
        # calibration was done after sigmoid, therefore only BCELoss() needed here:
        BCE = torch.nn.BCELoss()(torch.tensor(rescaled_pred), torch.tensor(result_true))
        JSD = - BCE.cpu().numpy() + np.log(2.)
        otp_str = "rescaled BCE loss of test set is {:.4f}, "+\
            "rescaled JSD of the two dists is {:.4f}"
        print(otp_str.format(BCE, JSD/np.log(2.)))
    return eval_acc, eval_auc, JSD/np.log(2.)

def calibrate_classifier(model, calibration_data):
    """ reads in calibration data and performs a calibration with isotonic regression"""
    model.eval()
    assert calibration_data is not None, ("Need calibration data for calibration!")
    for j, data_batch in enumerate(calibration_data):
        if args.save_mem:
            data_batch = data_batch[0].to(args.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = torch.sigmoid(output_vector).reshape(-1)
        target = target_vector.to(torch.float64)
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    result_true = result_true.cpu().numpy()
    result_pred = result_pred.cpu().numpy()
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6).fit(result_pred,
                                                                                      result_true)
    return iso_reg


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

def extract_shower_and_energy(given_file, which):
    """ reads .hdf5 file and returns samples and their energy """
    print("Extracting showers from {} file ...".format(which))
    shower = given_file['showers'][:]
    energy = given_file['incident_energies'][:]
    print("Extracting showers from {} file: DONE.\n".format(which))
    return shower, energy

def load_reference(filename):
    """ Load existing pickle with high-level features for reference in plots """
    print("Loading file with high-level features.")
    with open(filename, 'rb') as file:
        hlf_ref = pickle.load(file)
    return hlf_ref

def save_reference(ref_hlf, fname):
    """ Saves high-level features class to file """
    print("Saving file with high-level features.")
    #filename = os.path.splitext(os.path.basename(ref_name))[0] + '.pkl'
    with open(fname, 'wb') as file:
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
        if arg.x_scale == 'log':
            bins = np.logspace(np.log10(arg.min_energy),
                               np.log10(reference_class.GetElayers()[key].max()),
                               40)
        else:
            bins = 40
        counts_ref, bins, _ = plt.hist(reference_class.GetElayers()[key], bins=bins,
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

def plot_cell_dist(shower_arr, ref_shower_arr, arg):
    """ plots voxel energies across all layers """
    plt.figure(figsize=(6, 6))
    if arg.x_scale == 'log':
        bins = np.logspace(np.log10(arg.min_energy),
                           np.log10(ref_shower_arr.max()),
                           50)
    else:
        bins = 50

    counts_ref, _, _ = plt.hist(ref_shower_arr.flatten(), bins=bins,
                                label='reference', density=True, histtype='stepfilled',
                                alpha=0.2, linewidth=2.)
    counts_data, _, _ = plt.hist(shower_arr.flatten(), label='generated', bins=bins,
                                 histtype='step', linewidth=3., alpha=1., density=True)
    plt.title(r"Voxel energy distribution")
    plt.xlabel(r'$E$ [MeV]')
    plt.yscale('log')
    if arg.x_scale == 'log':
        plt.xscale('log')
    #plt.xlim(*lim)
    plt.legend(fontsize=20)
    plt.tight_layout()
    if arg.mode in ['all', 'hist-p', 'hist']:
        filename = os.path.join(arg.output_dir,
                                'voxel_energy_dataset_{}.png'.format(arg.dataset))
        plt.savefig(filename, dpi=300)
    if arg.mode in ['all', 'hist-chi', 'hist']:
        seps = separation_power(counts_ref, counts_data, bins)
        print("Separation power of voxel distribution histogram: {}".format(seps))
        with open(os.path.join(arg.output_dir,
                               'histogram_chi2_{}.txt'.format(arg.dataset)), 'a') as f:
            f.write('Voxel distribution: \n')
            f.write(str(seps))
            f.write('\n\n')
    plt.close()

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

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    source_file = h5py.File(args.input_file, 'r')
    check_file(source_file, args, which='input')

    particle = {'1-photons': 'photon', '1-pions': 'pion',
                '2': 'electron', '3': 'electron'}[args.dataset]
    # minimal readout per voxel, ds1: from Michele, ds2/3: 0.5 keV / 0.033 scaling factor
    args.min_energy = {'1-photons': 10, '1-pions': 10,
                       '2': 0.5e-3/0.033, '3': 0.5e-3/0.033}[args.dataset]

    hlf = HLF.HighLevelFeatures(particle,
                                filename='binning_dataset_{}.xml'.format(
                                    args.dataset.replace('-', '_')))
    shower, energy = extract_shower_and_energy(source_file, which='input')

    # get reference folder and name of file
    args.source_dir, args.reference_file_name = os.path.split(args.reference_file)
    print('Storing reference .pkl file in folder: {}'.format(args.source_dir))
    args.reference_file_name = os.path.splitext(args.reference_file_name)[0]

    reference_file = h5py.File(args.reference_file, 'r')
    check_file(reference_file, args, which='reference')

    reference_shower, reference_energy = extract_shower_and_energy(reference_file,
                                                                   which='reference')
    if os.path.exists(os.path.join(args.source_dir, args.reference_file_name + '.pkl')):
        print("Loading .pkl reference")
        reference_hlf = load_reference(os.path.join(args.source_dir,
                                                    args.reference_file_name + '.pkl'))
    else:
        print("Computing .pkl reference")
        reference_hlf = HLF.HighLevelFeatures(particle,
                                              filename='binning_dataset_{}.xml'.format(
                                                  args.dataset.replace('-', '_')))
        reference_hlf.Einc = reference_energy
        save_reference(reference_hlf,
                       os.path.join(args.source_dir, args.reference_file_name + '.pkl'))

    args.x_scale = 'log'

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
            save_reference(reference_hlf,
                           os.path.join(args.source_dir, args.reference_file_name + '.pkl'))
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
                save_reference(reference_hlf,
                               os.path.join(args.source_dir, args.reference_file_name + '.pkl'))

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
            save_reference(reference_hlf,
                           os.path.join(args.source_dir, args.reference_file_name + '.pkl'))

        print("Calculating high-level features for histograms: DONE.\n")

        if args.mode in ['all', 'hist-chi', 'hist']:
            with open(os.path.join(args.output_dir, 'histogram_chi2_{}.txt'.format(args.dataset)),
                      'w') as f:
                f.write('List of chi2 of the plotted histograms,'+\
                        ' see eq. 15 of 2009.03796 for its definition.\n')
        print("Plotting histograms ...")
        plot_histograms(hlf, reference_hlf, args)
        plot_cell_dist(shower, reference_shower, args)
        print("Plotting histograms: DONE. \n")

    if args.mode in ['all', 'cls-low', 'cls-high', 'cls-low-normed']:
        print("Calculating high-level features for classifier ...")
        hlf.CalculateFeatures(shower)
        hlf.Einc = energy

        if reference_hlf.E_tot is None:
            reference_hlf.CalculateFeatures(reference_shower)
            save_reference(reference_hlf,
                           os.path.join(args.source_dir, args.reference_file_name + '.pkl'))

        print("Calculating high-level features for classifer: DONE.\n")

        if args.mode in ['all', 'cls-low']:
            source_array = prepare_low_data_for_classifier(source_file, hlf, 0.,
                                                           normed=False)
            reference_array = prepare_low_data_for_classifier(reference_file, reference_hlf, 1.,
                                                              normed=False)
        elif args.mode in ['cls-low-normed']:
            source_array = prepare_low_data_for_classifier(source_file, hlf, 0.,
                                                           normed=True)
            reference_array = prepare_low_data_for_classifier(reference_file, reference_hlf, 1.,
                                                              normed=True)
        elif args.mode in ['cls-high']:
            source_array = prepare_high_data_for_classifier(source_file, hlf, 0.)
            reference_array = prepare_high_data_for_classifier(reference_file, reference_hlf, 1.)

        train_data, test_data, val_data = ttv_split(source_array, reference_array)

        # set up device
        args.device = torch.device('cuda:'+str(args.which_cuda) \
                                   if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        print("Using {}".format(args.device))

        # set up DNN classifier
        input_dim = train_data.shape[1]-1
        DNN_kwargs = {'num_layer':args.cls_n_layer,
                      'num_hidden':args.cls_n_hidden,
                      'input_dim':input_dim,
                      'dropout_probability':args.cls_dropout_probability}
        classifier = DNN(**DNN_kwargs)
        classifier.to(args.device)
        print(classifier)
        total_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

        print("{} has {} parameters".format(args.mode, int(total_parameters)))

        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)

        if args.save_mem:
            train_data = TensorDataset(torch.tensor(train_data))
            test_data = TensorDataset(torch.tensor(test_data))
            val_data = TensorDataset(torch.tensor(val_data))
        else:
            train_data = TensorDataset(torch.tensor(train_data).to(args.device))
            test_data = TensorDataset(torch.tensor(test_data).to(args.device))
            val_data = TensorDataset(torch.tensor(val_data).to(args.device))

        train_dataloader = DataLoader(train_data, batch_size=args.cls_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=args.cls_batch_size, shuffle=False)
        val_dataloader = DataLoader(val_data, batch_size=args.cls_batch_size, shuffle=False)

        train_and_evaluate_cls(classifier, train_dataloader, test_dataloader, optimizer, args)
        classifier = load_classifier(classifier, args)

        with torch.no_grad():
            print("Now looking at independent dataset:")
            eval_acc, eval_auc, eval_JSD = evaluate_cls(classifier, val_dataloader,
                                                        final_eval=True,
                                                        calibration_data=test_dataloader)
        print("Final result of classifier test (AUC / JSD):")
        print("{:.4f} / {:.4f}".format(eval_auc, eval_JSD))
        with open(os.path.join(args.output_dir, 'classifier_{}_{}.txt'.format(args.mode,
                                                                              args.dataset)),
                  'a') as f:
            f.write('Final result of classifier test (AUC / JSD):\n'+\
                    '{:.4f} / {:.4f}\n\n'.format(eval_auc, eval_JSD))



    # make plots next to each other
