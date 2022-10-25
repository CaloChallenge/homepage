# pylint: disable=invalid-name
""" helper file containing plotting functions to evaluate contributions to the
    Fast Calorimeter Challenge 2022.

    by C. Krause
"""

import os

import numpy as np
import matplotlib.pyplot as plt



def plot_layer_comparison(hlf_class, data, reference_class, reference_data, arg, show=False):
    """ plots showers of of data and reference next to each other, for comparison """
    num_layer = len(reference_class.relevantLayers)
    vmax = np.max(reference_data)
    layer_boundaries = np.unique(reference_class.bin_edges)
    for idx, layer_id in enumerate(reference_class.relevantLayers):
        plt.figure(figsize=(6, 4))
        reference_data_processed = reference_data\
            [:, layer_boundaries[idx]:layer_boundaries[idx+1]]
        reference_class._DrawSingleLayer(reference_data_processed,
                                         idx, filename=None,
                                         title='Reference Layer '+str(layer_id),
                                         fig=plt.gcf(), subplot=(1, 2, 1), vmax=vmax,
                                         colbar='None')
        data_processed = data[:, layer_boundaries[idx]:layer_boundaries[idx+1]]
        hlf_class._DrawSingleLayer(data_processed,
                                   idx, filename=None,
                                   title='Generated Layer '+str(layer_id),
                                   fig=plt.gcf(), subplot=(1, 2, 2), vmax=vmax, colbar='both')

        filename = os.path.join(arg.output_dir,
                                'Average_Layer_{}_dataset_{}.png'.format(layer_id, arg.dataset))
        plt.savefig(filename, dpi=300)
        if show:
            plt.show()
        plt.close()

def plot_Etot_Einc_discrete(hlf_class, reference_class, arg):
    """ plots Etot normalized to Einc histograms for each Einc in ds1 """
    # hardcode boundaries?
    bins = np.linspace(0.4, 1.4, 21)
    plt.figure(figsize=(10, 10))
    target_energies = 2**np.linspace(8, 23, 16)
    for i in range(len(target_energies)-1):
        if i > 3 and 'photons' in arg.dataset:
            bins = np.linspace(0.9, 1.1, 21)
        energy = target_energies[i]
        which_showers_ref = ((reference_class.Einc.squeeze() >= target_energies[i]) & \
                             (reference_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        which_showers_hlf = ((hlf_class.Einc.squeeze() >= target_energies[i]) & \
                             (hlf_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        ax = plt.subplot(4, 4, i+1)
        counts_ref, _, _ = ax.hist(reference_class.GetEtot()[which_showers_ref] /\
                                   reference_class.Einc.squeeze()[which_showers_ref],
                                   bins=bins, label='reference', density=True,
                                   histtype='stepfilled', alpha=0.2, linewidth=2.)
        counts_data, _, _ = ax.hist(hlf_class.GetEtot()[which_showers_hlf] /\
                                    hlf_class.Einc.squeeze()[which_showers_hlf], bins=bins,
                                    label='generated', histtype='step', linewidth=3., alpha=1.,
                                    density=True)
        if i in [0, 1, 2]:
            energy_label = 'E = {:.0f} MeV'.format(energy)
        elif i in np.arange(3, 12):
            energy_label = 'E = {:.1f} GeV'.format(energy/1e3)
        else:
            energy_label = 'E = {:.1f} TeV'.format(energy/1e6)
        ax.text(0.95, 0.95, energy_label, ha='right', va='top',
                transform=ax.transAxes)
        ax.set_xlabel(r'$E_{\text{tot}} / E_{\text{inc}}$')
        ax.xaxis.set_label_coords(1., -0.15)
        ax.set_ylabel('counts')
        ax.yaxis.set_ticklabels([])
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of Etot / Einc at E = {} histogram: {}".format(energy, seps))
        with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                  'a') as f:
            f.write('Etot / Einc at E = {}: \n'.format(energy))
            f.write(str(seps))
            f.write('\n\n')
        h, l = ax.get_legend_handles_labels()
    ax = plt.subplot(4, 4, 16)
    ax.legend(h, l, loc='center', fontsize=20)
    ax.axis('off')
    filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}_E_i.png'.format(arg.dataset))
    plt.savefig(filename, dpi=300)
    plt.close()

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
        seps = _separation_power(counts_ref, counts_data, bins)
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
            seps = _separation_power(counts_ref, counts_data, bins)
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
        plt.yscale('log')
        plt.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'ECEta_layer_{}_dataset_{}.png'.format(key,
                                                                           arg.dataset))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
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
        plt.yscale('log')
        plt.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'ECPhi_layer_{}_dataset_{}.png'.format(key,
                                                                           arg.dataset))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
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
        plt.yscale('log')
        plt.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'WidthEta_layer_{}_dataset_{}.png'.format(key,
                                                                              arg.dataset))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
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
        plt.yscale('log')
        plt.xlim(*lim)
        plt.legend(fontsize=20)
        plt.tight_layout()
        if arg.mode in ['all', 'hist-p', 'hist']:
            filename = os.path.join(arg.output_dir,
                                    'WidthPhi_layer_{}_dataset_{}.png'.format(key,
                                                                              arg.dataset))
            plt.savefig(filename, dpi=300)
        if arg.mode in ['all', 'hist-chi', 'hist']:
            seps = _separation_power(counts_ref, counts_data, bins)
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
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of voxel distribution histogram: {}".format(seps))
        with open(os.path.join(arg.output_dir,
                               'histogram_chi2_{}.txt'.format(arg.dataset)), 'a') as f:
            f.write('Voxel distribution: \n')
            f.write(str(seps))
            f.write('\n\n')
    plt.close()

def _separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts
    """
    hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()
