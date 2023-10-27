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

## User guide
### STAGE 1: Preparing the data
1) Create a new directory in **_sample-data_** with sample name - let's say "S1"
2) Within the "S1" directory, create a new directory "_raw_data"
3) Save your EDS elemental maps and SEM micrographs within "_raw_data" directory.
4) Naming convention: EDS maps must be saved as "eds-map-El" where "El" signifies element, for example : Al map will be named as "eds-map-Al", Fe map will be "eds-map-Fe"
5) Naming convention: Microstructure image must be saved as "micrograph-type" where "type" can be changed to anything that identifies the micrograph, for example: secondary image can be "micrograph-se", back scattered image can be "micrograph-bse"
6) Within the "S1" directory, create a new json file "_info_sample.json". This can be copied from the sample directory available in this repository.
7) Enter sample information in the "_info_sample.json" file. While updating this, use the file provided in this repository for reference.
   - **_sample_id_**: (Not mandatory) ID tag for sample
   - **_material_name_**: (Not mandatory) Sample material
   - **_material_process_state_**:(Not mandatory) Sample processing state
   - **_notes_**:(Not mandatory) Any notes on sample
   - **_eds_overall_composition: {elements}_**: (MANDATORY) The overall composition of the scanned region as determined by EDS software. Note: This is NOT the nominal composition of the sample. The format to enter this is: "El1": 20, "El2": 30, "El3": 50
   - **_eds_overall_composition: {units}_**: (MANDATORY) Units in which above composition was measured: "at_percent" or "wt_percent"
   - **_eds_map_elements_**: (MANDATORY) Elements for which EDS map has been saved in the sample data.
   - **_eds_map_unit_type_**: (MANDATORY) Unit type used for EDS maps: "at_percent" or "wt_percent"
   - **_eds_map_format_**: (MANDATORY) Extension of EDS map files: ".jpg" or ".png" or ".bmp"
   
### STAGE 2: Cropping the images
1) Open the "" notebook.


### STAGE 3: Analyzng and plotting the EDS data

