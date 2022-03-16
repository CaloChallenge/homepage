<!--## Welcome to the home of the Fast Calorimeter Simulation Challenge 2022!-->

<!-- ![img](Banner_grey.jpg) -->

<p style='text-align: justify;'>
Welcome to the home of the first-ever Fast Calorimeter Simulation Challenge! 
</p>

<p style='text-align: justify;'>
The purpose of this challenge is to spur the development and benchmarking of fast and high-fidelity calorimeter shower generation using deep learning methods. Currently, generating calorimeter showers of interacting particles (electrons, photons, pions, ...) using GEANT4 is a major computational bottleneck at the LHC, and it is forecast to overwhelm the computing budget of the LHC experiments in the near future. Therefore there is an urgent need to develop GEANT4 emulators that are both fast (computationally lightweight) and accurate. The LHC collaborations have been developing fast simulation methods for some time, and the hope of this challenge is to directly compare new deep learning approaches on common benchmarks. It is expected that participants will make use of cutting-edge techniques in generative modeling with deep learning, e.g. GANs, VAEs and normalizing flows. 
</p>

<p style='text-align: justify;'>
This challenge is modeled after two previous, highly successful data challenges in HEP &ndash; the <a href='https://arxiv.org/abs/1902.09914'>top tagging community challenge</a> and the <a href='https://arxiv.org/abs/2101.08320'>LHC Olympics 2020 anomaly detection challenge</a>. 
</p>

### Datasets

<p style='text-align: justify;'>
The challenge offers three datasets, ranging in difficulty from <q>easy</q> to <q>medium</q> to <q>hard</q>. The difficulty is set by the dimensionality of the calorimeter showers (the number layers and the number of voxels in each layer).
</p>
<p style='text-align: justify;'>
Each dataset has the same general format. The detector geometry consists of concentric cylinders with particles propagating along the z-axis. The detector is segmented along the z-axis into discrete layers. Each layer has bins along the radial direction and some of them have bins in the angle &alpha;. The number of layers and the number of bins in r and &alpha; is stored in the binning .xml files and will be read out by the HighLevelFeatures class of helper functions. The coordinates &Delta;&phi; and &Delta;&eta; correspond to the x- and y axis of the cylindrical coordinates. The image below shows a 3d view of a geometry with 3 layers, with each layer having 3 bins in radial and 6 bins in angular direction. The right image shows the front view of the geometry, as seen along the z axis.
</p>
<img src="https://calochallenge.github.io/homepage/coordsys.jpg" width="100%" align="center"/>
<p style='text-align: justify;'>
Each CaloChallenge dataset comes as one or more .hdf5 files that were written with python's h5py module using gzip compression. Within each file, there are two hdf5-datasets: <q>incident_energies</q> has the shape (num_events, 1) and contains the energy of the incoming particle in MeV, <q>showers</q> has the shape (num_events, num_voxels) and stores the showers, where the energy depositions of each voxel (in MeV) are flattened. The mapping of array index to voxel location is done at the order (radial bins, angular bins, layer), so the first entries correspond to the radial bins of the first angular slice in the first layer. Then, the radial bins of the next angular slice of the first layer follow, ... 
</p>

- <p style='text-align: justify;'> <b>Dataset 1</b> can be downloaded from <a href="https://doi.org/10.5281/zenodo.6234055">Zenodo with DOI /10.5281/zenodo.6234055</a>. It is based on the ATLAS GEANT4 open datasets that were published <a href="http://opendata-qa.cern.ch/record/15012">here</a>. There are two groups of datasets, one for photons and one for charged pions. Each set consists of 15 energies from 256 MeV up to 4 TeV produced in powers of two. Each dataset contains the voxelised shower information obtained from single particles  produced at the calorimeter surface in the &eta; range (0.2-0.25) and simulated in the ATLAS detector. 10k events are available in each sample with the exception of those at higher energies that have a lower statistics. These samples were used to train the corresponding two GANs presented in the AtlFast3 paper <a href="https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/SIMU-2018-04/">SIMU-2018-04</a> and in the FastCaloGAN note <a href="https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-SOFT-PUB-2020-006/">ATL-SOFT-PUB-2020-006</a>. The number of radial and angular bins varies from layer to layer and is also different for photons and pions, resulting in 368 voxels for photons and 533 for pions.</p>

- <p style='text-align: justify;'> <b>Dataset 2</b> (COMING SOON) consists of a single file with GEANT4-simulated showers of electrons with energies sampled from a log-uniform distribution ranging from 1 GeV to 1 TeV. The detector has a  concentric cylinder geometry with 45 layers, where each layer consists of active (silicon) and passive (tungesten) material. Each layer has 144 readout cells, 9 in radial and 16 in angular direction, yielding a total of 9x16x45 = 6480 voxels.  </p>

- <p style='text-align: justify;'> <b>Dataset 3</b> (COMING SOON). It consists of 3 files, each one contains 50k GEANT4-simulated eletron showers with energies sampled from a log-uniform distribution ranging from 1 GeV to 1 TeV. The detector geometry is similar to dataset 2, but has a much higher granularity. Each of the 45 layer has now 18 radial and 50 angular bins, totalling 18x50x45=40500 voxels. This dataset was produced using the <a href="https://gitlab.cern.ch/geant4/geant4/-/tree/master/examples/extended/parameterisations/Par04">Par04 Geant4 example</a>. </p>

<p style='text-align: justify;'>
Files containing the detector binning information for each dataset as well as Python scripts that load them can be found on our <a href="https://github.com/CaloChallenge/homepage/tree/main/code">Github page</a>. This <a href="https://github.com/CaloChallenge/homepage/blob/main/code/HighLevelFeatures.ipynb"> Jupyter notebook </a> shows how each dataset can be loaded, how the helper class is initialized with the binning.xml files, how high-level features can be computed, and how showers can be visualized. Further high-level features and histograms might be added in the next weeks.</p>


### Metrics

<p style='text-align: justify;'>

The overarching goal of the challenge is to train a generative model on the datasets provided and learn to sample from the conditional probability distribution $p(x|E)$, where $x$ are the voxel energy deposits and $E$ is the incident energy.
</p>

<p style='text-align: justify;'>

Participants will be scored using a variety of metrics. We will include more detailed descriptions of them in the coming months. Metrics will include:
</p>
- A binary classifier trained on <q>truth</q> GEANT4 vs. generated shower images.
- A binary classifier trained on a set of high-level features (like layer energies, shower shape variables).
- A chi-squared type measure derived from histogram differences of high-level features.
- Training time, calorimeter shower generation time and memory usage.
- Interpolation capabilities: leave out one energy in training and generate samples at that energy after training. 
<p style='text-align: justify;'>
It is expected that there will not necessarily be a single clear winner, but different methods will have their pros and cons. Scripts to perform the evaluation will eventually be made available on the <a href="https://github.com/CaloChallenge/homepage/tree/main/code">Github page</a>. 
</p>

### Timeline

<p style='text-align: justify;'>
The challenge will conclude approximately 1 month before the next ML4Jets conference (currently tentatively scheduled for the week of December 5, 2022). Results of the challenge will be presented at ML4Jets, and the challenge will culminate in a community paper documenting the various approaches and their outcomes. 
</p>

Please do not hesitate to ask questions: we will use the [ML4Jets slack channel](https://join.slack.com/t/ml4jets/shared_invite/enQtNDc4MjAzODE0NDIyLTU0MGIxNmZlY2E4MzY2YzEwNGI2MGI5MzJmMzEwODVjYWY4MDFhMzcyODYyMDViZTY4MTg2MWM2N2Y1YjBhOWM) to discuss technical questions related to this challenge. You are also encouraged to sign up for the <a href="https://groups.google.com/g/calochallenge"> Google groups mailing list </a> for infrequent announcements and communications.

Good luck!

_Michele Faucci Gianelli, Gregor Kasieczka, Claudius Krause, Ben Nachman, Dalila Salamani, David Shih and Anna Zaborowska_



<!---

You can use the [editor on GitHub](https://github.com/LHC-Olympics-2020/homepage/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/LHC-Olympics-2020/homepage/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

--->
