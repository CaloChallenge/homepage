import numpy as np
import math

import XMLHandler from XMLHandler

class HighFeatureFeatures:

  def __init__(self):
    
    xml = XMLHandler()
    self.bin_edges = xml.GetBinEdges()
    self.eta_all_layers, phi_all_layers = xml.GetEtaPhiAllLayers()
    self.relevantLayers = xml.GetRelevantLayers()
    self.layersBinnedInAlpha = xml.GetLayersWithBinningInAlpha()
    
  def calculate_EC(self, eta, phi, energy):
      eta_EC = (eta * energy).sum(axis=0)/energy.sum(axis=0)
      phi_EC = (phi * energy).sum(axis=0)/energy.sum(axis=0)
      return eta_EC, phi_EC

  def calculate_Widths(self, eta, phi, energy):
      eta_width = (eta * eta * energy).sum(axis=0)/energy.sum(axis=0)
      phi_width = (phi * phi * energy).sum(axis=0)/energy.sum(axis=0)
      return eta_width, phi_width
    
  def GetECandWidths(eta_layer, phi_layer, energy_layer):
    eta_EC = 0
    phi_EC = 0
    eta_width = 0
    phi_width = 0

    if (sum(energy_layer)!= 0):
      eta_EC , phi_EC = self.calculate_EC(eta_layer, phi_layer, energy_layer)
      eta_width , phi_width = self.calculate_Widths(eta_layer, phi_layer, energy_layer)
      # The following checks are needed to assure a positive argument to the sqrt, if there is very little energy things can go wrong
      if (eta_EC * eta_EC <= eta_width):
        eta_width = math.sqrt(eta_width - eta_EC * eta_EC) 
      if (phi_EC * phi_EC <= phi_width):
        phi_width = math.sqrt(phi_width - phi_EC * phi_EC)
        
    return eta_EC, phi_EC, eta_width, phi_width
    
  def EvaluateFeatures(self, data):
    self.E_tot = array( 'f', [ 0 ] )

    self.E_layers = {}
    for l in relevantLayers:
      E_layer = array( 'f', [ 0 ] )
      E_layers[l] = E_layer

    self.EC_etas = {}
    self.EC_phis = {}
    self.width_etas = {}
    self.width_phis = {}
    for l in layersBinnedInAlpha:
      EC_eta = array( 'f', [ 0 ] )
      EC_phi = array( 'f', [ 0 ] )
      width_eta = array( 'f', [ 0 ] )
      width_phi = array( 'f', [ 0 ] )
      EC_etas[l] = EC_eta
      EC_phis[l] = EC_phi
      width_etas[l] = width_eta
      width_phis[l] = width_phi
    
    for i in range (0, min(n_events,data.shape[0])):
      for l in relevantLayers:
        layer_energy = data[i,bin_edges[l]:bin_edges[l+1]].sum(axis=0)

        E_tot[0] += layer_energy
        E_layers[l][0] = layer_energy
                   
        for l in range(0,24):
          if l in layersBinnedInAlpha: 
            EC_etas[l][0], EC_phis[l][0], width_etas[l][0], width_phis[l][0] = GetECandWidths(eta_all_layers[l] , phi_all_layers[l], E_event[bin_edges[l]:bin_edges[l+1]])
                    
  def GetEtot(self):
    return self.E_tot
    
  def GetElayers(self):
    return self.E_layers    

  def GetECetas(self):
    return self.EC_etas
    
  def GetECphis(self):
    return self.EC_phis
    
  def GetWidthEtas(self):
    return self.width_etas
    
  def GetWidthPhis(self):
    return self.width_phis