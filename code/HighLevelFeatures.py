# pylint: disable=invalid-name
"""
    Class that handles the specific binning geometry based on the provided file
    and computes all relevant high-level features
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm as LN
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from XMLHandler import XMLHandler

class HighLevelFeatures:
    """ Computes all high-level features based on the specific geometry stored in the binning file
    """
    def __init__(self, particle, filename='binning.xml'):
        """ particle (str): particle to be considered
            filename (str): path/to/binning.xml of the specific detector geometry.
            particle is redundant, as it is also part of the binning file, however, it serves as a
            crosscheck to ensure the correct binning is used.
        """
        xml = XMLHandler(particle, filename=filename)
        self.bin_edges = xml.GetBinEdges()
        self.eta_all_layers, self.phi_all_layers = xml.GetEtaPhiAllLayers()
        self.relevantLayers = xml.GetRelevantLayers()
        self.layersBinnedInAlpha = xml.GetLayersWithBinningInAlpha()
        self.r_edges = [redge for redge in xml.r_edges if len(redge) > 1]
        self.num_alpha = [len(xml.alphaListPerLayer[idx][0]) for idx, redge in \
                          enumerate(xml.r_edges) if len(redge) > 1]
        self.E_tot = None
        self.E_layers = {}
        self.EC_etas = {}
        self.EC_phis = {}
        self.width_etas = {}
        self.width_phis = {}
        self.particle = particle

        self.num_voxel = []
        for idx, r_values in enumerate(self.r_edges):
            self.num_voxel.append((len(r_values)-1)*self.num_alpha[idx])

    def _calculate_EC(self, eta, phi, energy):
        eta_EC = (eta * energy).sum(axis=-1)/(energy.sum(axis=-1)+1e-16)
        phi_EC = (phi * energy).sum(axis=-1)/(energy.sum(axis=-1)+1e-16)
        return eta_EC, phi_EC

    def _calculate_Widths(self, eta, phi, energy):
        eta_width = (eta * eta * energy).sum(axis=-1)/(energy.sum(axis=-1)+1e-16)
        phi_width = (phi * phi * energy).sum(axis=-1)/(energy.sum(axis=-1)+1e-16)
        return eta_width, phi_width

    def GetECandWidths(self, eta_layer, phi_layer, energy_layer):
        """ Computes center of energy in eta and phi as well as their widths """
        eta_EC, phi_EC = self._calculate_EC(eta_layer, phi_layer, energy_layer)
        eta_width, phi_width = self._calculate_Widths(eta_layer, phi_layer, energy_layer)
        # The following checks are needed to assure a positive argument to the sqrt,
        # if there is very little energy things can go wrong
        eta_width = np.sqrt((eta_width - eta_EC**2).clip(min=0.))
        phi_width = np.sqrt((phi_width - phi_EC**2).clip(min=0.))
        return eta_EC, phi_EC, eta_width, phi_width

    def CalculateFeatures(self, data):
        """ Computes all high-level features for the given data """
        self.E_tot = data.sum(axis=-1)

        for l in self.relevantLayers:
            E_layer = data[:, self.bin_edges[l]:self.bin_edges[l+1]].sum(axis=-1)
            self.E_layers[l] = E_layer

        for l in self.relevantLayers:

            if l in self.layersBinnedInAlpha:
                self.EC_etas[l], self.EC_phis[l], self.width_etas[l], \
                    self.width_phis[l] = self.GetECandWidths(
                        self.eta_all_layers[l],
                        self.phi_all_layers[l],
                        data[:, self.bin_edges[l]:self.bin_edges[l+1]])

    def _DrawSingleLayer(self, data, layer_nr, filename, title=None, fig=None, subplot=(1, 1, 1),
                         vmax=None, colbar='alone'):
        """ draws the shower in layer_nr only """
        if fig is None:
            fig = plt.figure(figsize=(2, 2), dpi=200)
        num_splits = 400
        max_r = 0
        for radii in self.r_edges:
            if radii[-1] > max_r:
                max_r = radii[-1]
        radii = np.array(self.r_edges[layer_nr])
        if self.particle != 'electron':
            radii[1:] = np.log(radii[1:])
        theta, rad = np.meshgrid(2.*np.pi*np.arange(num_splits+1)/ num_splits, radii)
        pts_per_angular_bin = int(num_splits / self.num_alpha[layer_nr])
        data_reshaped = data.reshape(int(self.num_alpha[layer_nr]), -1)
        data_repeated = np.repeat(data_reshaped, (pts_per_angular_bin), axis=0)
        ax = fig.add_subplot(*subplot, polar=True)
        ax.grid(False)
        if vmax is None:
            vmax = data.max()
        pcm = ax.pcolormesh(theta, rad, data_repeated.T+1e-16, norm=LN(vmin=1e-2, vmax=vmax))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if self.particle == 'electron':
            ax.set_rmax(max_r)
        else:
            ax.set_rmax(np.log(max_r))
        if title is not None:
            ax.set_title(title)
        #wdth = str(len(self.r_edges)*100)+'%'
        if colbar == 'alone':
            axins = inset_axes(fig.get_axes()[-1], width='100%',
                               height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
                               bbox_transform=fig.get_axes()[-1].transAxes,
                               borderpad=0)
            cbar = plt.colorbar(pcm, cax=axins, fraction=0.2, orientation="horizontal")
            cbar.set_label(r'Energy (MeV)', y=0.83, fontsize=12)
        elif colbar == 'both':
            axins = inset_axes(fig.get_axes()[-1], width='200%',
                               height="15%", loc='lower center',
                               bbox_to_anchor=(-0.625, -0.2, 1, 1),
                               bbox_transform=fig.get_axes()[-1].transAxes,
                               borderpad=0)
            cbar = plt.colorbar(pcm, cax=axins, fraction=0.2, orientation="horizontal")
            cbar.set_label(r'Energy (MeV)', y=0.83, fontsize=12)
        elif colbar == 'None':
            pass
        #if title is not None:
        #    plt.gcf().suptitle(title)
        if filename is not None:
            plt.savefig(filename, facecolor='white')
        #return fig

    def _DrawShower(self, data, filename, title):
        """ Draws the shower in all layers """
        if self.particle == 'electron':
            figsize = (10, 20)
        else:
            figsize = (len(self.relevantLayers)*2, 3)
        fig = plt.figure(figsize=figsize, dpi=200)
        # to smoothen the angular bins (must be multiple of self.num_alpha):
        num_splits = 400
        layer_boundaries = np.unique(self.bin_edges)
        max_r = 0
        for radii in self.r_edges:
            if radii[-1] > max_r:
                max_r = radii[-1]
        vmax = data.max()
        for idx, layer in enumerate(self.relevantLayers):
            radii = np.array(self.r_edges[idx])
            if self.particle != 'electron':
                radii[1:] = np.log(radii[1:])
            theta, rad = np.meshgrid(2.*np.pi*np.arange(num_splits+1)/ num_splits, radii)
            pts_per_angular_bin = int(num_splits / self.num_alpha[idx])
            data_reshaped = data[layer_boundaries[idx]:layer_boundaries[idx+1]].reshape(
                int(self.num_alpha[idx]), -1)
            data_repeated = np.repeat(data_reshaped, (pts_per_angular_bin), axis=0)
            if self.particle == 'electron':
                ax = plt.subplot(9, 5, idx+1, polar=True)
            else:
                ax = plt.subplot(1, len(self.r_edges), idx+1, polar=True)
            ax.grid(False)
            pcm = ax.pcolormesh(theta, rad, data_repeated.T+1e-16, norm=LN(vmin=1e-2, vmax=vmax))
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if self.particle == 'electron':
                ax.set_rmax(max_r)
            else:
                ax.set_rmax(np.log(max_r))
            ax.set_title('Layer '+str(layer))
        if self.particle == 'electron':
            axins = inset_axes(fig.get_axes()[-3], width="500%",
                               height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
                               bbox_transform=fig.get_axes()[-3].transAxes,
                               borderpad=0)
        else:
            wdth = str(len(self.r_edges)*100)+'%'
            axins = inset_axes(fig.get_axes()[len(self.r_edges)//2], width=wdth,
                               height="15%", loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1),
                               bbox_transform=fig.get_axes()[len(self.r_edges)//2].transAxes,
                               borderpad=0)
        cbar = plt.colorbar(pcm, cax=axins, fraction=0.2, orientation="horizontal")
        cbar.set_label(r'Energy (MeV)', y=0.83, fontsize=12)
        if title is not None:
            plt.gcf().suptitle(title)
        if filename is not None:
            plt.savefig(filename, facecolor='white')
        else:
            plt.show()
        plt.close()

    def GetEtot(self):
        """ returns total energy of the showers """
        return self.E_tot

    def GetElayers(self):
        """ returns energies of the showers deposited in each layer """
        return self.E_layers

    def GetECEtas(self):
        """ returns dictionary of centers of energy in eta for each layer """
        return self.EC_etas

    def GetECPhis(self):
        """ returns dictionary of centers of energy in phi for each layer """
        return self.EC_phis

    def GetWidthEtas(self):
        """ returns dictionary of widths of centers of energy in eta for each layer """
        return self.width_etas

    def GetWidthPhis(self):
        """ returns dictionary of widths of centers of energy in phi for each layer """
        return self.width_phis

    def DrawAverageShower(self, data, filename=None, title=None):
        """ plots average of provided showers """
        self._DrawShower(data.mean(axis=0), filename=filename, title=title)

    def DrawSingleShower(self, data, filename=None, title=None):
        """ plots all provided showers after each other """
        ret = []
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        for num, shower in enumerate(data):
            if filename is not None:
                local_name, local_ext = os.path.splitext(filename)
                local_name += '_{}'.format(num) + local_ext
            else:
                local_name = None
            self._DrawShower(shower, filename=local_name, title=title)

    def DrawHistoEtot(self, filename=None):
        raise NotImplementedError()

    def DrawHistoElayers(self, filename=None):
        raise NotImplementedError()

    def DrawHistoECEtas(self, filename=None):
        raise NotImplementedError()

    def DrawHistoECPhis(self, filename=None):
        raise NotImplementedError()

    def DrawHistoWidthEtas(self, filemane=None):
        raise NotImplementedError()

    def DrawHistoWidthPhis(self, filename=None):
        raise NotImplementedError()
