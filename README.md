## Welcome to the home of the Fast Calorimeter Simulation Challenge 2022!

<!-- ![img](Banner_grey.jpg) -->

<p style='text-align: justify;'>
This is the homepage for the Fast Calorimeter Simulation Data Challenge. The purpose of this challenge is to spur the development and benchmarking of fast and high-fidelity calorimeter shower generation. Currently, generating calorimeter showers of elementary particles (electrons, photons, pions, ...) using GEANT4 is a major computational bottleneck at the LHC, and it is forecast to overwhelm the computing budget of the LHC in the near future. Therefore there is an urgent need to develop GEANT4 emulators that are both fast (computationally lightweight) and accurate. It is expected that participants will make use of cutting-edge techniques in generative modeling with deep learning, e.g. GANs, VAEs and normalizing flows. 
</p>

<p style='text-align: justify;'>
This challenge is modeled after previous, highly successful data challenges in HEP &ndash; the <a href='https://arxiv.org/abs/1902.09914'>top tagging community challenge</a> and the <a href='https://arxiv.org/abs/2101.08320'>LHC Olympics 2020 anomaly detection challenge</a>. 
</p>

### Datasets

<p style='text-align: justify;'>
We expect to release up to three datasets during the course of the challenge, ranging in difficulty from <q>easy</q> to <q>medium</q> to <q>hard</q>. The difficulty is set by the dimensionality of the calorimeter showers (the number layers and the number of voxels in each layer). The <q>hard</q> Dataset 3 also includes incoming particles at different angles.
</p>

<p style='text-align: justify;'>
- Dataset 1: The ATLAS GEANT4 open datasets [link](http://opendata-qa.cern.ch/record/15012). There are two groups of datasets, one for charged pions and one for photons. Each set consists of 15 csv files corresponsing to 15 energies from 256 MeV up to 4TeV produced in powers of two. Each dataset contains the voxelised shower information obtained from single particles  produced at the calorimeter surface in the η range (0.2-0.25) and simulated in the ATLAS detector. 10k events are available in each sample with the exception of those at higher energies that have a lower statistics. These samples were used to train the corresponding two GANs presented in the AtlFast3 paper [SIMU-2018-04](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/SIMU-2018-04/) and in the FastCaloGAN note [ATL-SOFT-PUB-2020-006](https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-SOFT-PUB-2020-006/).
</p>

<p style='text-align: justify;'>
- Dataset 2: the calorimeter used to produce this GEANT4 dataset is a setup of concentric cylinders (layers). Each layer consists of active (silicon) and passive (tungesten) material. The number of readout cells is **RxPxZ=9x16x45=6480**, representing the cylindrical segmentation **(<sub>r;</sub><sub>&phi;</sub><sub>;z</sub>)**. The dataset consists of eletron particles with energies ranging from 1 GeV to 1024 GeV (in powers of 2). 
</p>

<p style='text-align: justify;'>
- Dataset 3 : the calorimeter used to produce this GEANT4 dataset is a also a setup of concentric cylinders of silicon and tungesten material. The number of readout cells is **RxPxZ=18x50x45=40500**, representing the cylindrical segmentation **(<sub>r;</sub><sub>&phi;</sub><sub>;z</sub>)**. The dataset consists of eletron particles with energies ranging from 1 GeV to 1024 GeV (in powers of 2) and incident angles ranging from 50 to 90&deg; in a step of 10&deg; (90&deg; angle indicates a particle entering the detector perpendicularly). This dataset was produced using the [Par04 Geant4 example](https://gitlab.cern.ch/geant4/geant4/-/tree/master/examples/extended/parameterisations/Par04).
</p>

### Metrics

<p style='text-align: justify;'>
Participants will be scored using a variety of metrics. These will include a binary classifier trained on <q>truth</q> GEANT4 vs. generated shower images, a binary classifier trained on a set of high level features (layer energies, shower shape variables), and metrics derived from histogram differences and chi-squared type measures. Finally, methods will be judged based on their training time, calorimeter shower generation time and memory usage. It is expected that there will not necessarily be a single clear winner, but different methods will have their pros and cons. A link to the full set of metrics will be provided here [link]. 
</p>

### Timeline

<p style='text-align: justify;'>
The challenge will conclude approximately 1 month before the next ML4Jets conference (currently tentatively scheduled for the week of December 5, 2022). Results of the challenge will be presented at ML4Jets, and the challenge will culminate in a community paper documenting the various approaches and their outcomes. 
</p>

Please do not hesitate to ask questions: we will use the [ML4Jets slack channel](https://join.slack.com/t/ml4jets/shared_invite/enQtNDc4MjAzODE0NDIyLTU0MGIxNmZlY2E4MzY2YzEwNGI2MGI5MzJmMzEwODVjYWY4MDFhMzcyODYyMDViZTY4MTg2MWM2N2Y1YjBhOWM) to discuss technical questions related to this challenge. You are also encouraged to sign up for the mailing list [GOOGLE GROUP]
for infrequent announcements and communications.

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
