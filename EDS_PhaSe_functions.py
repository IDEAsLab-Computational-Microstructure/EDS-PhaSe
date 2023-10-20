import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import os
import json
import pickle

import ipywidgets as ipyw

import helper_functions as f_helpers


SEP = os.sep


# -----------------------------
def load_eds_data(sample, source):
    
    """
    function to load the raw data (eds maps and micrographs) of a sample
    INPUT:
        sample: (str) name of directory containing sample information
        source: (str) Source of eds files (RAW_Data, INITIAL_Crop_Data, FINAL_Crop_Data)
    OUTPUT:
        dict_sample_info: (dict) dictionary containing sample information
        dict_eds: (dict) dictionary with color eds maps as 3D numpy arrays
        dict_ms: (dict) dictionary with micrographs
    """
    
    sample_dir_path = f"sample-data{SEP}{sample}{SEP}" # path to sample directory
    
    # load sample info json file into a dictionary
    sample_info_path = sample_dir_path + "_info_sample.json"
    
    with open(sample_info_path, "r") as f:
        dict_sample_info = json.load(f)
    
    eds_el_list = dict_sample_info["eds_map_elements"].split(" ") # read list of elements in sample data
    dict_sample_info["eds_map_elements"] = eds_el_list # update dict key with list of elements
    
    
    dict_eds = {} # empty dict to store color raw eds maps
    eds_map_format = dict_sample_info["eds_map_format"] # read format in which raw eds data is stored
    
    if source == "RAW_Data":
        data_dir_path = f"{sample_dir_path}_raw-data{SEP}"
        
    if source == "INITIAL_Crop_Data":
        data_dir_path = f"{sample_dir_path}INITIAL_Crop_Data{SEP}"
        
    if source == "FINAL_Crop_Data":
        data_dir_path = f"{sample_dir_path}FINAL_Crop_Data{SEP}"
        
    for el in eds_el_list:
        eds_el_path = data_dir_path + f"eds-map-{el}{eds_map_format}" # path to element eds map
        
        if eds_map_format in [".png", ".jpg", ".jpeg"]:
            eds_el_im_color = cv2.imread(eds_el_path) # load 3-channel color eds image (openCV loads as BGR)
            eds_el_im_color = np.array(cv2.cvtColor(eds_el_im_color, cv2.COLOR_BGR2RGB)) # Convert from BGR to RGB
            dict_eds[el] = eds_el_im_color
        
        if eds_map_format == ".xlsx":
            eds_el_df = pd.read_excel(eds_el_path, header=None)
            dict_eds[el] = np.array(eds_el_df)
        
    
    ref_eds_map_shape = dict_eds[eds_el_list[0]].shape
    print("Checking if all eds maps have same dimension... ", end="", flush=True)
    
    if all(dict_eds[el].shape == ref_eds_map_shape for el in eds_el_list):
        print("DONE. ALL OK.", flush=True)
    else:
        print("DONE. FAILED. Check elemental image dimensions. Or, this may be corrected while cropping.", flush=True)
    
    dict_ms = {} # empty dict to store raw micrographs

    for filename in os.listdir(data_dir_path):
        if "micrograph" in filename:
            micrograph_path = data_dir_path + filename
            micrograph_type = filename.split("-")[1].split(".")[0]
            micrograph_im_color = cv2.imread(micrograph_path)
            micrograph_im_color = np.array(cv2.cvtColor(micrograph_im_color, cv2.COLOR_BGR2RGB))
            dict_ms[micrograph_type] = micrograph_im_color
    
    
    return (dict_sample_info, dict_eds, dict_ms)



# -----------------------------
def run_interactive_crop():

    """runs an interactive ipywidget program for cropping EDS maps and micrograph images
    """
    
    def interactive_crop(sample, cropStage, cropType, imageKey, resizePercent, figSize, rowI, rowF, colI, colF):
        
        """function linked with the interactive widgets
        """
    
        # Creating cache to avoid repetitive calculations for same sample
        if cropStage == "INITIAL_CROP": source = "RAW_Data"
        if cropStage == "FINAL_CROP": source = "INITIAL_Crop_Data"
        
        if sample not in list(dict_cache_for_crop[source].keys()):
            dict_sample_info, dict_eds, dict_ms = load_eds_data(sample, source)
            dict_cache_for_crop[source][sample] = {"info":dict_sample_info, "eds":dict_eds, "ms":dict_ms}
        else:
            dict_sample_info = dict_cache_for_crop["RAW_Data"][sample]["info"]
            dict_eds = dict_cache_for_crop[source][sample]["eds"]
            dict_ms = dict_cache_for_crop[source][sample]["ms"]

        global eds_map_format
        eds_map_format = dict_sample_info["eds_map_format"] # read format in which raw eds data is stored

        # Update 'W_imageKey' widget options
        if cropType == "EDS Map":    
            W_imageKey.options = list(dict_eds.keys()) + [""]
            img_to_use = dict_eds[imageKey]

        if cropType == "Micrograph":
            W_imageKey.options = list(dict_ms.keys()) + [""]
            img_to_use = dict_ms[imageKey]
        
        # getting start/end values for row and columns from interactive widgets
        global rowI_val, rowF_val, colI_val, colF_val
        rowI_val, rowF_val, colI_val, colF_val = rowI, rowF, colI, colF
        
        # creating global variables that will be used in crop function
        global sample_val, cropStage_val, cropType_val, eds_el_list, imageKey_val, resizePercent_val
        resizePercent_val = resizePercent
        sample_val, cropStage_val, cropType_val, imageKey_val = sample, cropStage, cropType, imageKey
        eds_el_list = list(dict_eds.keys())
        
        # Setting min/max values of sliders for row/column values
        W_rowF.value, W_colF.value = min(img_to_use.shape[0], rowF), min(img_to_use.shape[1], colF)
        W_rowI.min, W_rowI.max = 0, img_to_use.shape[0]
        W_rowF.min, W_rowF.max = 0, img_to_use.shape[0]
        W_colI.min, W_colI.max = 0, img_to_use.shape[1]
        W_colF.min, W_colF.max = 0, img_to_use.shape[1]
        
        # Crop the image and display it
        crop_map = img_to_use[rowI:rowF, colI:colF]
        fig_size = (figSize, figSize)
        plt.figure(figsize=(fig_size[0]*2, fig_size[1]*2))
        #plt.imshow(crop_map, cmap="inferno", interpolation="none")
        plt.title(f"{cropType}: {imageKey}", fontsize=15)
        plt.imshow(crop_map)

        print(f"Cropped Image: rows [{rowI}:{rowF}]; columns [{colI}:{colF}]")
        print("Image shape:", crop_map.shape)
        resize_dim = tuple([int(i*resizePercent/100) for i in crop_map.shape[:2]])
        print("Resized shape:", resize_dim)

        return None


    sample_list = os.listdir(f"sample-data{SEP}") # get list of all samples available

    # creating 
    global dict_cache_for_crop
    dict_cache_for_crop = {"RAW_Data": {}, "INITIAL_Crop_Data": {}, "FINAL_Crop_Data": {}}
    
    
    for i in sample_list:
        dict_cache_for_crop["INITIAL_Crop_Data"][i] = {"info":{}, "eds":{}, "ms":{}}
        dict_cache_for_crop["FINAL_Crop_Data"][i] = {"info":{}, "eds":{}, "ms":{}}
    
    
    W_sample = ipyw.Dropdown(options=sample_list, value=sample_list[0], description="Sample")
    W_cropStage = ipyw.Dropdown(options=["INITIAL_CROP", "FINAL_CROP"], value="INITIAL_CROP", description="Crop Stage")
    W_cropType = ipyw.Dropdown(options=["EDS Map", "Micrograph"], value="EDS Map", description="Crop Type")
    W_imageKey = ipyw.Dropdown(description="Use Image", value=None)
    W_resizePercent = ipyw.IntSlider(min=1, max=100, step=1, value=100, description='Shrink Percentage')
    W_figSize = ipyw.IntSlider(min=2, max=10, step=1, value=4, description='Figure Size')
    W_rowI = ipyw.IntSlider(description='Row start')
    W_rowF = ipyw.IntSlider(min=0, max=2000, value=2000, description='Row end')
    W_colI = ipyw.IntSlider(description='Column start')
    W_colF = ipyw.IntSlider(min=0, max=2000, value=2000, description='Column end')

    ipyw.interact(interactive_crop,
                  sample=W_sample,
                  cropStage = W_cropStage,
                  cropType = W_cropType,
                  imageKey = W_imageKey,
                  resizePercent = W_resizePercent,
                  figSize = W_figSize,
                  rowI = W_rowI,
                  rowF = W_rowF,
                  colI = W_colI,
                  colF = W_colF)

    # Creating button to save
    W_saveButton = ipyw.Button(description="Save Crop Images")
    W_output = ipyw.Output()
    display(W_saveButton, W_output)

    def on_saveButton_clicked(b):
    
        """funtion to save cropped images. It is executed when 'W_saveButton' button is clicked.
        """
        
        print("", end="", flush=True)
        W_output.clear_output()

        if cropStage_val == "INITIAL_CROP":
            destination = "INITIAL_Crop_Data"; source = "RAW_Data"
        if cropStage_val == "FINAL_CROP":
            destination = "FINAL_Crop_Data"; source = "INITIAL_Crop_Data"

        with W_output:

            sample_dir_path = f"sample-data{SEP}{sample_val}{SEP}" # path to sample directory
            crop_save_dir = f"{sample_dir_path}{destination}{SEP}" # path to crop save directory

            if not os.path.exists(crop_save_dir):
                os.mkdir(crop_save_dir)

            if cropType_val == "EDS Map":
                for el in eds_el_list:
                    eds_map_crop = dict_cache_for_crop[source][sample_val]["eds"][el][rowI_val:rowF_val, colI_val:colF_val]
                    crop_save_path = crop_save_dir + f"eds-map-{el}{eds_map_format}"

                    if eds_map_format in [".png", ".jpg", ".jpeg"]:
                    
                        eds_resize_dim = tuple(reversed([int(i*resizePercent_val/100) for i in eds_map_crop.shape[:2]]))
                        eds_map_crop_resized = cv2.resize(eds_map_crop, eds_resize_dim, interpolation= cv2.INTER_AREA)
                        cv2.imwrite(crop_save_path, cv2.cvtColor(eds_map_crop_resized, cv2.COLOR_RGB2BGR)) #convert to BGR and then write (because cv2 writes as BGR)
                        dict_cache_for_crop[destination][sample_val]["eds"][el] = eds_map_crop_resized
                    
                    if eds_map_format == ".xlsx":
                        eds_map_crop_df = pd.DataFrame(eds_map_crop)
                        dict_cache_for_crop[destination][sample_val]["eds"][el] = np.array(eds_map_crop_df)
                        eds_map_crop_df.to_excel(crop_save_path, header=None, index=False)

            if cropType_val == "Micrograph":
                ms_crop = dict_cache_for_crop[source][sample_val]["ms"][imageKey_val][rowI_val:rowF_val, colI_val:colF_val]

                #if cropStage_val == "INITIAL_CROP":
                
                if eds_map_format in [".png", ".jpg", ".jpeg"]:
                    eds_crop_dim = dict_cache_for_crop[destination][sample_val]["eds"][eds_el_list[0]].shape[:2]
                
                if eds_map_format == ".xlsx":
                    eds_crop_dim = dict_cache_for_crop[destination][sample_val]["eds"][eds_el_list[0]].shape
                
                ms_crop = cv2.resize(ms_crop, tuple(reversed(eds_crop_dim)), interpolation= cv2.INTER_AREA)
                
                dict_cache_for_crop[destination][sample_val]["ms"][imageKey_val] = ms_crop
                crop_save_path = crop_save_dir + f"micrograph-{imageKey_val}.png"
                cv2.imwrite(crop_save_path, cv2.cvtColor(ms_crop, cv2.COLOR_RGB2BGR)) #convert to BGR and then write (because cv2 writes as BGR)

            print("Done. Cropped images saved.")

    W_saveButton.on_click(on_saveButton_clicked)
    
    return None



# -----------------------------
def eds_get_single_channel(dict_eds, method="max_variance"):
    
    """
    function to convert multi-channel eds images into single channel
    INPUT:
        dict_eds: (dict)
        method: (str) method to be used for conversion. currently available methods:
            -"max_variance": channel that shows maximum variance
            -"max_intensity": channel that shows maximum intensity value
            -"openCV_RGB2GRAY": using openCV 'COLOR_RGB2GRAY' method for color to grayscale
    OUTPUT:
        dict_eds_single_ch: (dict) dictionary with single channel (2D) eds maps
    """
    
    eds_el_list = list(dict_eds.keys())
    
    dict_eds_single_ch = {}
    
    for el in eds_el_list:
        if len(dict_eds[el].shape) > 2:
            if method == "max_variance":
                dict_eds_single_ch[el] = f_helpers.max_variance_channel(dict_eds[el])
            if method == "max_intensity":
                dict_eds_single_ch[el] = f_helpers.max_intensity_channel(dict_eds[el])
            if method == "openCV_RGB2GRAY":
                dict_eds_single_ch[el] = f_helpers.RGB2GRAY_img(dict_eds[el])

        else:
            dict_eds_single_ch[el] = dict_eds[el]
        
    return (dict_eds_single_ch)
    
    

# -----------------------------------
def scale_eds_maps_relative_to_overall_comp(dict_sample_info, dict_eds_single_ch):
    
    """
    function to convert single channel intensity eds maps into composition maps
    INPUT:
        dict_sample_info: (dict) dictionary containing sample information
        dict_eds_single_ch: (dict) dictionary with single channel (2D) eds maps
    OUTPUT:
        dict_eds_scaled: (dict) dictionary with scaled eds maps
        eds_overall_comp_unit: (str) type relative to which scaling was done
            -"at_percent": scaling was done relative to atomic percent
            -"wt_percent": scaling was done relative to weight percent
    """
    
    # extract overall composition from sample information dictionary
    eds_overall_comp = dict_sample_info["eds_overall_composition"]["elements"]
    eds_overall_comp_unit = dict_sample_info["eds_overall_composition"]["units"]
    
    # get overall composition in both at_percent and wt_percent through conversion functions
    if eds_overall_comp_unit == "at_percent":
        eds_overall_comp_at_percent = eds_overall_comp
        eds_overall_comp_wt_percent = f_helpers.convert_comp_at_to_wt(eds_overall_comp_at_percent)
    
    if eds_overall_comp_unit == "wt_percent":
        eds_overall_comp_wt_percent = eds_overall_comp
        eds_overall_comp_at_percent = f_helpers.convert_comp_wt_to_at(eds_overall_comp_wt_percent)
        
    el_list = list(dict_eds_single_ch.keys())
    
    dict_eds_scaled = {} # dictionary to store scale eds maps
    
    for el in el_list:
        sum_single_ch = np.sum(dict_eds_single_ch[el]) # sum of all element signals at each pixel

        if eds_overall_comp_unit == "at_percent":
            el_at_percent = eds_overall_comp_at_percent[el]
            mod_factor = el_at_percent/sum_single_ch # modification factor for scaling

        if eds_overall_comp_unit == "wt_percent":
            el_wt_percent = eds_overall_comp_wt_percent[el]
            mod_factor = el_wt_percent/sum_single_ch # modification factor for scaling

        dict_eds_scaled[el] = np.array(dict_eds_single_ch[el] * mod_factor, dtype=float)
    
    return (dict_eds_scaled, eds_overall_comp_unit)
    
    
    
# -----------------------------------
def convert_eds_at_to_wt(dict_eds_comp_at):
    
    """
    function to convert eds composition maps from atomic percent to weight percent
    INPUT:
        dict_eds_comp_at: (dict) dictionary with atomic percent elemental maps
    OUTPUT:
        dict_eds_comp_wt: (dict) dictionary with weight percent elemental maps
    """
    
    db_el_path = "_program-data" + SEP + "database-element.json"

    with open(db_el_path, "r") as f:
        dict_db_el = json.load(f)
        
    el_list = list(dict_eds_comp_at.keys())
    
    array_pixel_wt_sum = np.zeros(shape=dict_eds_comp_at[el_list[0]].shape)
    for el in el_list:
        el_wt = dict_db_el[el]["atomic-weight"] 
        eds_el_wt = dict_eds_comp_at[el] * el_wt
        array_pixel_wt_sum = np.add(array_pixel_wt_sum, eds_el_wt)
    
    dict_eds_comp_wt = {}

    for el in el_list:
        el_wt = dict_db_el[el]["atomic-weight"] 
        eds_el_wt = dict_eds_comp_at[el] * el_wt
        eds_el_wt = 100 * np.divide(eds_el_wt, array_pixel_wt_sum)
        dict_eds_comp_wt[el] = eds_el_wt
        
        
    return (dict_eds_comp_wt)


# -----------------------------------
def convert_eds_wt_to_at(dict_eds_comp_wt):
    
    """
    function to convert eds composition maps from atomic percent to weight percent
    INPUT:
        dict_eds_comp_wt: (dict) dictionary with weight percent elemental maps
    OUTPUT:
        dict_eds_comp_at: (dict) dictionary with atomic percent elemental maps
    """

    db_el_path = "_program-data" + SEP + "database-element.json"

    with open(db_el_path, "r") as f:
        dict_db_el = json.load(f)
        
    el_list = list(dict_eds_comp_wt.keys())
    
    array_pixel_moles_sum = np.zeros(shape=dict_eds_comp_wt[el_list[0]].shape)
    for el in el_list:
        el_wt = dict_db_el[el]["atomic-weight"] 
        eds_el_moles = dict_eds_comp_wt[el] / el_wt
        array_pixel_moles_sum = np.add(array_pixel_moles_sum, eds_el_moles)
    
    dict_eds_comp_at = {};

    for el in el_list:
        el_wt = dict_db_el[el]["atomic-weight"] 
        eds_el_moles = dict_eds_comp_wt[el] / el_wt
        eds_el_at = 100 * np.divide(eds_el_moles, array_pixel_moles_sum)
        dict_eds_comp_at[el] = eds_el_at
        
    
    return (dict_eds_comp_at)



# -----------------------------------
def get_comp_from_scaled_eds(dict_eds_scaled, eds_overall_comp_unit):

    """
    function to convert scaled eds maps into atomic_percent and wt_percent maps
    INPUT:
        dict_eds_scaled: (dict) dictionary with scaled eds maps
        eds_overall_comp_unit: (str) type relative to which scaling was done
            -"at_percent": scaling was done relative to atomic percent
            -"wt_percent": scaling was done relative to weight percent
    OUTPUT:
        dict_eds_comp: (dict) dictionary with atomic percent and weight percent elemental maps
    """
    
    el_list = list(dict_eds_scaled.keys())
    array_pixel_sum = np.zeros(shape=dict_eds_scaled[el_list[0]].shape)
    
    for el in el_list: 
        array_pixel_sum = np.add(array_pixel_sum, dict_eds_scaled[el])
    
    dict_eds_comp = {"at_percent":{},
                     "wt_percent":{}}
    
    for el in el_list:
        eds_comp_el = 100 * (np.asarray(dict_eds_scaled[el], dtype=float)/array_pixel_sum)
        dict_eds_comp[eds_overall_comp_unit][el] = eds_comp_el
        
    if eds_overall_comp_unit == "at_percent":
        dict_eds_comp["wt_percent"] = convert_eds_at_to_wt(dict_eds_comp["at_percent"])
    
    if eds_overall_comp_unit == "wt_percent":
        dict_eds_comp["at_percent"] = convert_eds_wt_to_at(dict_eds_comp["wt_percent"])


    return (dict_eds_comp)
    
    
    
# -----------------------------------
def plot_dict(dict_input, nCols=6, cmap="inferno", clim=(0, 255)):
    
    """
    function to plot values stored in all keys in any given dictionary
    INPUT:
        dict_input: (dict) input dictionary
        nCols: (int) number of columns in final plot
        cmap: (string) colormap to be used (e.g. "inferno", "viridis", "rainbow", etc.)
        clim: (tuple) range of colorbar used for color mapping
    OUTPUT:
        None
    """
    
    fig_size = (10,8)
    fs = 28 #fontsize
    
    # plot all keys
    keys_list = list(dict_input.keys())
    nPlots = len(keys_list) #number of total plots
    nRows = round(nPlots/nCols) + 1
    
    fig = plt.figure(figsize=(fig_size[0]*nCols, fig_size[1]*nRows))
    for (key, i_plot) in zip(keys_list, range(1, nPlots+1)):

        plt.subplot(nRows, nCols, i_plot)
        plt.title("\n'%s'"%(key), fontsize=fs + 2)
        im = plt.imshow(dict_input[key], cmap=cmap, interpolation="none")
        
        clim_low, clim_high = clim[0], clim[1]
        plt.clim(clim_low, clim_high)
        cb = plt.colorbar(orientation="vertical", shrink=.7)
        cb.ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.show()
    
    return None
    
    
    

# -----------------------------------
def binary_diff(dict_eds_el_input):
    
    """
    function to calculate the pixel wise difference for element pairs
    INPUT:
        dict_eds_el_input: (dict) input dictionarty with EDS elemental maps
    OUTPUT:
        dict_binary_diff: (dict) dictionary with difference maps
    """
    
    el_list = list(dict_eds_el_input.keys())
    dict_binary_diff = {}

    for elA in el_list:
        dict_binary_diff[elA] = {}
    
        for elB in el_list:
            if elA != elB:
                binary = "%s-%s"%(elA, elB)
                dict_binary_diff[elA][binary] = dict_eds_el_input[elA] - dict_eds_el_input[elB]


    return (dict_binary_diff)
    


# -----------------------------------
def binary_absdiff(dict_eds_el_input):
    
    """
    function to calculate the pixel wise absolute difference for element pairs
    INPUT:
        dict_eds_el_input: (dict) input dictionarty with EDS elemental maps
    OUTPUT:
        dict_binary_absdiff: (dict) dictionary with absolute difference maps
    """
    
    el_list = list(dict_eds_el_input.keys())
    dict_binary_absdiff = {}

    for elA in el_list:
        dict_binary_absdiff[elA] = {}
    
        for elB in el_list:
            if elA != elB:
                binary = "%s-%s"%(elA, elB)
                dict_binary_absdiff[elA][binary] = np.abs(dict_eds_el_input[elA] - dict_eds_el_input[elB])


    return (dict_binary_absdiff)



# -----------------------------------
def binary_pdt(dict_eds_el_input):
    
    """
    function to calculate the pixel wise product for element pairs
    INPUT:
        dict_eds_el_input: (dict) input dictionarty with EDS elemental maps
    OUTPUT:
        dict_binary_pdt: (dict) dictionary with product maps
    """
    
    el_list = list(dict_eds_el_input.keys())
    dict_binary_pdt = {}

    for elA in el_list:
        dict_binary_pdt[elA] = {}
    
        for elB in el_list:
            if elA != elB:
                binary = "%s-%s"%(elA, elB)
                product = np.multiply(dict_eds_el_input[elA], dict_eds_el_input[elB])
                product_norm = (product-np.min(product))/(np.max(product)-np.min(product))
                dict_binary_pdt[elA][binary] = product_norm


    return (dict_binary_pdt)



# -----------------------------------
def binary_order_param(dict_eds_el_input):
    
    """
    function to calculate the pixel wise difference for element pairs
    INPUT:
        dict_eds_el_input: (dict) input dictionarty with EDS elemental maps
    OUTPUT:
        dict_order_param: (dict) dictionary with order parameter maps
    """
    
    el_list = list(dict_eds_el_input.keys())
    dict_order_param = {}

    for elA in el_list:
        dict_order_param[elA] = {}
    
        for elB in el_list:
            if elA != elB:
                binary = "%s-%s"%(elA, elB)
                product = np.multiply(dict_eds_el_input[elA], dict_eds_el_input[elB])
                uniform_dist_pdt = np.mean(dict_eds_el_input[elA]) * np.mean(dict_eds_el_input[elB])
                order_param = (product/uniform_dist_pdt - 1)
                dict_order_param[elA][binary] = order_param


    return (dict_order_param)



# -----------------------------------
def create_EDS_PhaSe_markerMapsCache(dict_EDS_PhaSe, save_cache=True):
    
    """
    function to create master dictionary with all binary parameter maps from EDS atomic and weight percent maps
    INPUT:
        dict_EDS_PhaSe: (dict) EDS dictionary with atomic and weight percent maps
    OUTPUT:
        dict_EDS_PhaSe_markerMaps: (dict) dictionary with binary parameter maps for both atomic and weight percent
    """
    
    dict_EDS_PhaSe_markerMaps = {
        "at_percent":{},
        "wt_percent":{}
    }
    
    # Create weight percent binary maps
    dict_EDS_PhaSe_markerMaps["wt_percent"]["order_parameter"] = binary_order_param(dict_EDS_PhaSe["eds"]["wt_percent"])
    dict_EDS_PhaSe_markerMaps["wt_percent"]["difference"] = binary_diff(dict_EDS_PhaSe["eds"]["wt_percent"])
    dict_EDS_PhaSe_markerMaps["wt_percent"]["abs_difference"] = binary_absdiff(dict_EDS_PhaSe["eds"]["wt_percent"])
    dict_EDS_PhaSe_markerMaps["wt_percent"]["product"] = binary_pdt(dict_EDS_PhaSe["eds"]["wt_percent"])
    
    # Create atomic percent binary maps
    dict_EDS_PhaSe_markerMaps["at_percent"]["order_parameter"] = binary_order_param(dict_EDS_PhaSe["eds"]["at_percent"])
    dict_EDS_PhaSe_markerMaps["at_percent"]["difference"] = binary_diff(dict_EDS_PhaSe["eds"]["at_percent"])
    dict_EDS_PhaSe_markerMaps["at_percent"]["abs_difference"] = binary_absdiff(dict_EDS_PhaSe["eds"]["at_percent"])
    dict_EDS_PhaSe_markerMaps["at_percent"]["product"] = binary_pdt(dict_EDS_PhaSe["eds"]["at_percent"])
    
    return (dict_EDS_PhaSe_markerMaps)
    
    
# -----------------------------------
def create_EDS_PhaSe_cache(sample, channel_extraction_method="max_variance", save_cache=True):
    
    """
    function to analyze and create cache with EDS_PhaSe analyzed data
    INPUT:
        sample: (str) name of directory containing sample information
        channel_extraction_method: (str) method to be used for extraction of single phase channel.
            -"max_variance": use channel that shows maximum variance
            -"max_intensity": use channel that shows maximum intensity value
            -"openCV_RGB2GRAY": using openCV 'COLOR_RGB2GRAY' method for color to grayscale
    OUTPUT:
        dict_EDS_PhaSe: (dict) dictionary containing all final data after EDS_PhaSe analysis.
    """
    
    dict_sample_info, dict_eds, dict_ms = load_eds_data(sample, source="FINAL_Crop_Data")
    dict_eds_single_ch = eds_get_single_channel(dict_eds, method=channel_extraction_method)
    dict_eds_scaled, eds_overall_comp_unit = scale_eds_maps_relative_to_overall_comp(dict_sample_info, dict_eds_single_ch)
    dict_eds_comp = get_comp_from_scaled_eds(dict_eds_scaled, eds_overall_comp_unit)
    
    dict_EDS_PhaSe = {
        "info": dict_sample_info,
        "ms": dict_ms,
        "eds": {
            "at_percent": dict_eds_comp["at_percent"],
            "wt_percent": dict_eds_comp["wt_percent"],
            "pixel_raw": dict_eds_single_ch,
            "pixel_scaled": dict_eds_scaled
            }
        }
    
    
    dict_EDS_PhaSe_markerMaps = create_EDS_PhaSe_markerMapsCache(dict_EDS_PhaSe)
    
    if save_cache == True:
        data_cache_name = "EDS_PhaSe_data.pkl"
        data_cache_path = f"sample-data{SEP}{sample}{SEP}{data_cache_name}"
        with open(data_cache_path, 'wb') as file:
            pickle.dump(dict_EDS_PhaSe, file)
            
        markerMaps_cache_name = "EDS_PhaSe_markerMaps.pkl"
        markerMaps_cache_path = f"sample-data{SEP}{sample}{SEP}{markerMaps_cache_name}"
        with open(markerMaps_cache_path, 'wb') as file:
            pickle.dump(dict_EDS_PhaSe_markerMaps, file)
          
          
    return (dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps)
    
    
    
# -----------------------------------
def load_EDS_PhaSe_cache(sample):

    """
    function to load EDS_PhaSe cache containing analysis results
    INPUT:
        sample: (str) name of directory containing sample information
    OUTPUT:
        dict_EDS_PhaSe: (dict) dictionary containing all final data after EDS_PhaSe analysis.
    """
    
    data_cache_name = "EDS_PhaSe_data.pkl"
    data_cache_path = f"sample-data{SEP}{sample}{SEP}{data_cache_name}"
    
    markerMaps_cache_name = "EDS_PhaSe_markerMaps.pkl"
    markerMaps_cache_path = f"sample-data{SEP}{sample}{SEP}{markerMaps_cache_name}"
    
    if os.path.exists(data_cache_path) and os.path.exists(markerMaps_cache_path):
        
        with open(data_cache_path, 'rb') as file:
            dict_EDS_PhaSe = pickle.load(file)
        
        with open(markerMaps_cache_path, 'rb') as file:
            dict_EDS_PhaSe_markerMaps = pickle.load(file)
        
    else:
        print("'EDS_PhaSe data' CACHE NOT FOUND. New calculation initiated...", end="", flush=True)
        dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps = create_EDS_PhaSe_cache(sample)
        print("DONE.", flush=True)
        
    
    return (dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps)
    
    
    
# -----------------------------------
def interactive_plot_markerMaps(dict_EDS_PhaSe_markerMaps):
    
    """
    function to launch interactively widget to display binary parameter maps
    INPUT:
        dict_EDS_PhaSe_markerMaps: (dict) dictionary with binary parameter maps for both atomic and weight percent
    OUTPUT:
        None
    """
    
    print("\n\nVisualize marker maps created by EDS_Phase analysis.")
    map_unit_list = list(dict_EDS_PhaSe_markerMaps.keys())
    parameters_list = list(dict_EDS_PhaSe_markerMaps["at_percent"].keys())
    cmap_list = ["seismic", "PiYG", "PRGn", "inferno", "viridis", "plasma", "magma", "cividis"]
    
    def plot_markerMaps(map_unit, parameter, nCols_input, cmap_input, clim_input):
        
        print("Visualization initiated...", end="", flush=True)
        dict_to_use = dict_EDS_PhaSe_markerMaps[map_unit][parameter]
        clow, chigh = float(clim_input.split(",")[0]), float(clim_input.split(",")[1])
        
        for el in list(dict_to_use.keys()):
            el_dict_to_use = dict_to_use[el]
            plot_dict(el_dict_to_use, nCols=nCols_input, cmap=cmap_input, clim=(clow, chigh))
            plt.show()
    
        print("DONE.")
    ipyw.interact_manual(plot_markerMaps, map_unit=map_unit_list, parameter=parameters_list,
                         nCols_input=6, cmap_input=cmap_list, clim_input = "-2,2")
                         
    return None
    
    

# -----------------------------------
def plot_dict_distribution(dict_input, x_label="x"):

    """
    function to plot the distribution of values in elemental maps in input dictionary
    INPUT:
        dict_input: (dict) input dictionary with EDS elemental maps
    OUTPUT:
        None
    """
    
    fig_size = (10,8)
    fs = 28 # fontsize

    plt.figure(figsize=fig_size)
    for key in list(dict_input.keys()):
        a = dict_input[key]
        sns.kdeplot(a.flatten(), shade=True, label=key)
    plt.title("Freq. density plot", fontsize=fs)
    plt.xlabel(x_label, fontsize=fs)
    plt.ylabel("frequency density", fontsize=fs)
    plt.xticks(fontsize=fs-2); plt.yticks(fontsize=fs-2)
    plt.legend(fontsize=fs-10)
    plt.show()



# -----------------------------------
def interactive_plot_edsMaps(dict_EDS_PhaSe):
    
    """
    function to launch interactive widget to display transformed eds maps
    INPUT:
        dict_EDS_PhaSe: (dict) dictionary containing all final data after EDS_PhaSe analysis.
    OUTPUT:
        None
    """
    
    print("\n\nVisualize transformed EDS maps created by EDS_Phase analysis.")
    
    plot_type_list = ["Element Maps", "Element distribution"]
    map_type_list = list(dict_EDS_PhaSe["eds"].keys())
    cmap_list = ["seismic", "PiYG", "PRGn", "inferno", "viridis", "plasma", "magma", "cividis"]
    
    def plot_edsMaps(plot_type, map_type, nCols_input, cmap_input, clim_input):
        
        print("Visualization initiated...", end="", flush=True)
        dict_to_use = dict_EDS_PhaSe["eds"][map_type]
        
        if plot_type == "Element Maps":
            clow, chigh = float(clim_input.split(",")[0]), float(clim_input.split(",")[1])
            plot_dict(dict_to_use, nCols=nCols_input, cmap=cmap_input, clim=(clow, chigh))
            plt.show()
            
        if plot_type == "Element distribution":
            plot_dict_distribution(dict_to_use, x_label=map_type)
            
        print("DONE.")
        
    ipyw.interact_manual(plot_edsMaps, plot_type=plot_type_list, map_type=map_type_list,
                         nCols_input=6, cmap_input=cmap_list, clim_input = "0,50")
                         
    return None
    
    

# -----------------------------------
def mask_from_parameter(dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps, sys, map_unit, parameter, operator, threshold, col_value):
    
    """
    function to create a mask based on threshold value of binary map properties
    INPUT:
        dict_EDS_PhaSe: (dict) dictionary containing all final data after EDS_PhaSe analysis.
        dict_EDS_PhaSe_markerMaps: (dict) dictionary with binary parameter maps for both atomic and weight percent
        sys: (str) binary system to use (e.g.- "Al-Cr", "Co-Fe")
        map_unit: (str) "at_percent" or "wt_percent"
        parameter: (str) parameter to be used ("order_parameter", "difference", "abs_difference", "product")
        operator: (str) operator to apply threshold ("show_above_threshold", "show_below_threshold")
        col_value: (float) value assigned in the mask; range [0, 1]
    OUTPUT:
        mask_im: (numpy array) mask based on input parameters
    """
    dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps
    if parameter == "composition":
        dict_to_use = dict_EDS_PhaSe["eds"][map_unit]
    else:
        dict_to_use = dict_EDS_PhaSe_markerMaps[map_unit][parameter]
    if len(sys.split("-")) == 1: ref_im = dict_to_use[sys]
    if len(sys.split("-")) == 2: ref_im = dict_to_use[sys.split("-")[0]][sys]
    
    mask_im = np.empty(shape=ref_im.shape)
    
    for row in range(ref_im.shape[0]):
        for col in range(ref_im.shape[1]):

            if operator == "show_above_threshold":
                if ref_im[row,col] >= float(threshold):
                    mask_im[row,col] = float(col_value)
                else:
                    mask_im[row,col] = np.nan

            if operator == "show_below_threshold":
                if ref_im[row,col] < float(threshold):
                    mask_im[row,col] = float(col_value)
                else:
                    mask_im[row,col] = np.nan
                        
    return (mask_im)

    
    
# -----------------------------------
def interactive_mask_from_parameter(dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps):
    
    """
    funtion to launch interactive widget for creating masks based on binary parameters
    INPUT:
        dict_EDS_PhaSe: (dict) dictionary containing all final data after EDS_PhaSe analysis.
        dict_EDS_PhaSe_markerMaps: (dict) dictionary with binary parameter maps for both atomic and weight percent
    OUTPUT:
        None
    """
    
    print("\n\nCreate masks on microstructure using 'EDS_PhaSe' calculated markers.")
    
    ms_list = list(dict_EDS_PhaSe["ms"].keys())
    map_unit_list = list(dict_EDS_PhaSe_markerMaps.keys())
    parameters_list = list(dict_EDS_PhaSe_markerMaps["at_percent"].keys()) + ["composition"]
    operator_list = ["show_above_threshold", "show_below_threshold"];

    def plot_mask(microstructure, sys, map_unit, parameter, operator, threshold, col_value, alpha, cmap_to_use):
        
        """
        function to plot the mask in interactive widget
        """
        mask_im = mask_from_parameter(dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps, sys, map_unit, parameter, operator, threshold, col_value)
        
        fig_size = (10,10)
        plt.figure(figsize=fig_size)
        ms_img_to_use = dict_EDS_PhaSe["ms"][microstructure]
        plt.imshow(ms_img_to_use, cmap="gray", alpha=1, interpolation="none")
        plt.imshow(mask_im, cmap=cmap_to_use, alpha=float(alpha), interpolation="none")
        plt.xticks(fontsize=15); plt.yticks(fontsize=15)
        plt.clim(0,1)
        plt.show()
        
        return None
        
    ipyw.interact(plot_mask, microstructure=ms_list, sys="Cr-Al", map_unit=map_unit_list, parameter=parameters_list,
             operator = operator_list, threshold="0.3", col_value="0.9", alpha="0.6", cmap_to_use="rainbow")
             
    return None
    
    
    
# -----------------------------------
def phase_analyze(dict_phase_mask, segment_im, dict_EDS_PhaSe):
    
    """
    function to analyze the phase segmentation
    INPUT:
        dict_phase_mask: (dict) dictionary with individual masks
        segment_im: (numpy array) segmented image
        dict_EDS_PhaSe: (dict) dictionary with eds maps (at_percent and wt_percent)
    OUTPUT:
        dict_phase_fracs: (dict) dictionary with phase fractions (area or volume)
        dict_phase_comp: (dict) dictionary with phase compositions
    """
    
    phase_names_list = list(dict_phase_mask.keys())
    n_phase = len(phase_names_list)
    el_list = list(dict_EDS_PhaSe["eds"]["at_percent"].keys())
    
    # calculate phase fractions
    phase_pix_counts = [np.count_nonzero(~np.isnan(dict_phase_mask[phase])) for phase in phase_names_list]
    phase_fracs = [np.around(count/np.sum(phase_pix_counts), 4) for count in phase_pix_counts]
    
    # calculate composition of phases
    dict_phase_fracs = {}
    dict_phase_comp = {"at_percent": {},
                       "wt_percent": {}
            }
    
    for (i, phase_name) in zip(range(0, n_phase), phase_names_list):
        
        dict_phase_fracs[phase_name] = phase_fracs[i]
        dict_phase_comp["at_percent"][phase_name] = {}
        dict_phase_comp["wt_percent"][phase_name] = {}
        
        phase_pix_count = phase_pix_counts[i]
        mask = np.nan_to_num(dict_phase_mask[phase_name], nan=0)
        mask[np.where(mask!=0)] = 1
        
        for el in el_list:
            el_eds_at = dict_EDS_PhaSe["eds"]["at_percent"][el]
            el_eds_wt = dict_EDS_PhaSe["eds"]["wt_percent"][el]
            
            el_at_total = np.sum(np.multiply(mask, el_eds_at))
            el_wt_total = np.sum(np.multiply(mask, el_eds_wt))
            
            dict_phase_comp["at_percent"][phase_name][el] = np.around(el_at_total/phase_pix_count, 2)
            dict_phase_comp["wt_percent"][phase_name][el] = np.around(el_wt_total/phase_pix_count, 2)
    
    print("Phase vol fractions: ")
    print(dict_phase_fracs)
    
    print("\nPhase compositions (at_percent):")
    for phase_name in list(dict_phase_comp["at_percent"].keys()):
        print(f"{phase_name}: {dict_phase_comp['at_percent'][phase_name]}")
        
    print("\nPhase compositions (wt_percent):")
    for phase_name in list(dict_phase_comp["wt_percent"].keys()):
        print(f"{phase_name}: {dict_phase_comp['wt_percent'][phase_name]}")
    
    
    
    return (dict_phase_fracs, dict_phase_comp)
    
    
    
# -----------------------------------
def interactive_phase_segmentation(dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps):
    
    """
    function to create masks and segment image
    INPUT:
        dict_EDS_PhaSe: (dict) dictionary containing all final data after EDS_PhaSe analysis.
        dict_EDS_PhaSe_markerMaps: (dict) dictionary with binary parameter maps for both atomic and weight percent
    OUTPUT:
        None
    """
    
    print("\n\nCreate final masks with user-defined parameters.")
    
    
    def phase_segmentation(sys_list, map_unit_list, parameter_list,
                           operator_list, threshold_list, col_list,
                           plot, cmap, clim_input):
        
        sys_values = sys_list.replace(" ","").split(",")
        map_unit_values = map_unit_list.replace(" ","").split(",")
        parameter_values = parameter_list.replace(" ","").split(",")
        operator_values = operator_list.replace(" ","").split(",")
        threshold_values = [float(i) for i in threshold_list.replace(" ","").split(",")]
        col_values = [float(i) for i in col_list.replace(" ","").split(",")]
        clim = tuple([float(i) for i in clim_input.replace(" ","").split(",")])
        
        
        dict_phase_mask = {}
        for i in range(0, len(col_values)-1):
                       
            phase_mask = mask_from_parameter(dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps,
                                             sys=sys_values[i], map_unit=map_unit_values[i],
                                             parameter=parameter_values[i], operator=operator_values[i],
                                             threshold=threshold_values[i], col_value=col_values[i])

            dict_phase_mask[f"mask_{i+1}:{sys_values[i]}"] = phase_mask

        phase_names_list = list(dict_phase_mask.keys())
        mask_shape = dict_phase_mask[phase_names_list[0]].shape
        phase_last_mask = np.empty(shape=mask_shape)
        segment_im = np.empty(shape=mask_shape)

        for row in range(phase_last_mask.shape[0]):
            for col in range(phase_last_mask.shape[1]):

                bool_to_fill = True # bool to decide if current has to be filled or not

                # if any phase is found at pixel, bool_to_fill turns to False and loop breaks
                for i in range(0, len(phase_names_list)):
                    if np.isnan(dict_phase_mask[phase_names_list[i]][row, col]):
                        continue
                    else:
                        segment_im[row, col] = col_values[i]
                        bool_to_fill = False
                        break

                if bool_to_fill == True:
                    phase_last_mask[row, col] = col_values[-1]
                    segment_im[row, col] = col_values[-1]
                else:
                    phase_last_mask[row, col] = np.nan

        dict_phase_mask[f"mask_{len(col_values)}"] = phase_last_mask

        if plot == True:
            plot_dict(dict_phase_mask, nCols=len(dict_phase_mask.keys()), cmap=cmap, clim=clim)

            # Plot overall segmented image
            fig_size = (20,15)
            plt.figure(figsize=fig_size)
            plt.imshow(segment_im, cmap=cmap, interpolation="none", clim=clim)
            plt.colorbar(shrink=0.7)
            plt.show()
            
        phase_analyze(dict_phase_mask, segment_im, dict_EDS_PhaSe)
                
        
    ipyw.interact_manual(phase_segmentation, sys_list="", map_unit_list="", parameter_list="",
                         operator_list="", threshold_list="", col_list="",
                         plot=True, cmap="viridis", clim_input="0,1")
                         
    return None
    
    
    
# -----------------------------------
def run_interactive_EDS_PhaSe_analysis():
    
    sample_list = os.listdir(f"sample-data{SEP}")

    W_cache_OR_new = ipyw.RadioButtons(options=["LOAD CACHE IF AVAILABLE", "RESET & CALCULATE AGAIN"], value="LOAD CACHE IF AVAILABLE", description="Use Cache / Do new calculation ?")
    W_sample = ipyw.Dropdown(options=sample_list, value=sample_list[0], description="Sample")
    W_AnalyzeButton = ipyw.Button(description="Analyze EDS Data")
    W_AnalyzeOutput = ipyw.Output()
    display(W_cache_OR_new, W_sample, W_AnalyzeButton, W_AnalyzeOutput)
    
    
    def on_AnalyzeButton_clicked(b):

        """funtion to analyze EDS data. It is executed when 'W_AnalyzeOutput' button is clicked.
        """

        W_AnalyzeOutput.clear_output()

        with W_AnalyzeOutput:

            global dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps
            if W_cache_OR_new.value == "RESET & CALCULATE AGAIN":
                dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps = create_EDS_PhaSe_cache(W_sample.value)
                print("'EDS_PhaSe data' and 'EDS_PhaSe marker maps' generated. Available for use.")


            if W_cache_OR_new.value == "LOAD CACHE IF AVAILABLE":
                dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps = load_EDS_PhaSe_cache(W_sample.value)
                print("'EDS_PhaSe data' and 'EDS_PhaSe marker maps' loaded. Available for use.")
            
            interactive_plot_edsMaps(dict_EDS_PhaSe)
            interactive_plot_markerMaps(dict_EDS_PhaSe_markerMaps)
            interactive_mask_from_parameter(dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps)
            interactive_phase_segmentation(dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps)
            
    W_AnalyzeButton.on_click(on_AnalyzeButton_clicked)
    
    return None
    
    
# -----------------------------------
def interactive_crop_region_composition(dict_EDS_PhaSe):
        
    def crop_region_composition(cropType, imageKey, figSize, rowI, rowF, colI, colF):

        W_imageKey.options = list(dict_EDS_PhaSe[cropType].keys()) + [""]
        img_to_use = dict_EDS_PhaSe[cropType][W_imageKey.value]

        # Setting min/max values of sliders for row/column values
        #W_rowI.value, W_colI.value = 0, 0
        W_rowF.value, W_colF.value = min(img_to_use.shape[0], rowF), min(img_to_use.shape[1], colF)
        W_rowI.min, W_rowI.max = 0, img_to_use.shape[0]
        W_rowF.min, W_rowF.max = 0, img_to_use.shape[0]
        W_colI.min, W_colI.max = 0, img_to_use.shape[1]
        W_colF.min, W_colF.max = 0, img_to_use.shape[1]

        crop_map = img_to_use[rowI:rowF, colI:colF]
        fig_size = (figSize, figSize)
        plt.figure(figsize=(fig_size[0]*2, fig_size[1]*2))
        #plt.imshow(crop_map, cmap="inferno", interpolation="none")
        plt.title(f"{cropType}: {imageKey}", fontsize=15)
        plt.imshow(crop_map)

        print(f"Cropped Image: rows [{rowI}:{rowF}]; columns [{colI}:{colF}]")
        print("Image shape:", crop_map.shape)

        crop_n_pix = crop_map.shape[0] * crop_map.shape[1]
        dict_crop_comp = {"at_percent": {},
                          "wt_percent": {}}

        for el in dict_EDS_PhaSe["eds"]["at_percent"].keys():
            crop_el_at_total = np.sum(dict_EDS_PhaSe["eds"]["at_percent"][el][rowI:rowF, colI:colF])
            dict_crop_comp["at_percent"][el] = np.around(crop_el_at_total/crop_n_pix, 2)

            crop_el_wt_total = np.sum(dict_EDS_PhaSe["eds"]["wt_percent"][el][rowI:rowF, colI:colF])
            dict_crop_comp["wt_percent"][el] = np.around(crop_el_wt_total/crop_n_pix, 2)

        print("\nCrop Region composition (at_percent):")
        print(dict_crop_comp["at_percent"])

        print("\nCrop Region composition (wt_percent):")
        print(dict_crop_comp["wt_percent"])
    
    
    W_cropType = ipyw.Dropdown(options=["eds", "ms"], value="ms", description="Crop Type")
    W_imageKey = ipyw.Dropdown(description="Use Image", value=None)
    W_figSize = ipyw.IntSlider(min=2, max=10, step=1, value=4, description='Figure Size')
    W_rowI = ipyw.IntSlider(description='Row start')
    W_rowF = ipyw.IntSlider(min=0, max=2000, value=2000, description='Row end')
    W_colI = ipyw.IntSlider(description='Column start')
    W_colF = ipyw.IntSlider(min=0, max=2000, value=2000, description='Column end')

    ipyw.interact(crop_region_composition,
                  cropType = W_cropType,
                  imageKey = W_imageKey,
                  figSize = W_figSize,
                  rowI = W_rowI,
                  rowF = W_rowF,
                  colI = W_colI,
                  colF = W_colF)
                  
    return None

    
                
# -----------------------------------
def run_interactive_region_composition():
    
    sample_list = os.listdir(f"sample-data{SEP}")

    W_cache_OR_new = ipyw.RadioButtons(options=["LOAD CACHE IF AVAILABLE", "RESET & CALCULATE AGAIN"], value="LOAD CACHE IF AVAILABLE", description="Use Cache / Do new calculation ?")
    W_sample = ipyw.Dropdown(options=sample_list, value=sample_list[0], description="Sample")
    W_AnalyzeButton = ipyw.Button(description="Analyze EDS Data")
    W_AnalyzeOutput = ipyw.Output()
    display(W_cache_OR_new, W_sample, W_AnalyzeButton, W_AnalyzeOutput)

    def on_AnalyzeButton_clicked(b):

            """funtion to analyze EDS data. It is executed when 'W_AnalyzeOutput' button is clicked.
            """

            W_AnalyzeOutput.clear_output()

            with W_AnalyzeOutput:

                global dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps
                if W_cache_OR_new.value == "RESET & CALCULATE AGAIN":
                    dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps = create_EDS_PhaSe_cache(W_sample.value)
                    print("'EDS_PhaSe data' and 'EDS_PhaSe marker maps' generated. Available for use.")


                if W_cache_OR_new.value == "LOAD CACHE IF AVAILABLE":
                    dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps = load_EDS_PhaSe_cache(W_sample.value)
                    print("'EDS_PhaSe data' and 'EDS_PhaSe marker maps' loaded. Available for use.")
                
                interactive_crop_region_composition(dict_EDS_PhaSe)
                
    W_AnalyzeButton.on_click(on_AnalyzeButton_clicked)
    
    return None