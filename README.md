# EDS-PhaSe
### Publication details:
[EDS-PhaSe: Phase segmentation and analysis from EDS elemental map images using markers of elemental segregation]() (to be linked soon)
- **Authors** : [Dishant Beniwal](https://github.com/d-beniwal) [^1], V Shivam [^2], O Palasyuk [^3], MJ Kramer [^3], G Phanikumar [^4], Pratik K. Ray [^1]
- **Journal** : Metallography, Microstructure, and Analysis
- **DOI** : [to be updated soon] Article is accepted but not available yet
[^1]: Department of Metallurgical and Materials Engineering, Indian Institute of Technology Ropar, Rupnagar 140001, Punjab, India
[^2]: Materials Engineering Division, CSIR-National Metallurgical Laboratory, Jamshedpur – 831007, Jharkhand, India
[^3]: Ames National Laboratory, US-DOE, Ames, IA – 50011, USA
[^4]: Department of Metallurgical and Materials Engineering, Indian Institute of Technology Madras, Chennai – 600036, Tamil Nadu, India

### About **EDS-PhaSe**
**EDS-PhaSe** performs quantitative analysis from EDS elemental maps and provides an interactive workflow to identify, segment and analyze phases present in any given microstructure. **EDS-PhaSe** is implemented through jupyter notebooks embedded with interactive widgets.

## Contents
- **__program-data_**: directory containing element database used by the code.
- **_sample-data_**: directory where user data (EDS elemental maps and SEM micrographs) have to be saved. For each sample, a new directory has to be created within this directory.
- **_EDS_PhaSe_Interactive.ipynb_**: jupyter notebook with interactive workflow for cropping and analyzing the data.
- **_EDS_PhaSe_functions.py_**: python script containing core functions utilised by EDS-PhaSe.
- **_helper_functions.py_**: python script containing helper functions utilized by EDS-PhaSe.
- **_J1.1_Region_Composition_Interactive.ipynb_**: jupyter notebook to get composition of specific regions from the overall microstructure.
