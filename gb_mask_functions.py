import cv2
import numpy as np
import pandas as pd

import EDS_PhaSe_functions as EDS_PhaSe

import os
        
        
SEP = os.sep



# -----------------------------
def create_fills_and_masks_around_gb(sample, max_px=100):
    """creates gb masks and fills around the grain boundary. These are saved as .npz files
    Args:
        sample (str): sample name to be analyzed; dir with same name must be present in sample_data
        max_px (int, default=100): max number of pixels around gb that will be included in masks
    Returns:
        None
    """

    sample_dir_path = "sample-data" + SEP + sample + SEP # path to sample directory
    gb_mask_dir_name = "gb_masks"
    gb_mask_dir_path = sample_dir_path + gb_mask_dir_name + SEP

    print("Loading input grain boundary mask... ", end="")
    # Load gb mask
    for filename in os.listdir(gb_mask_dir_path):
        if ".png" in filename:
            gb_mask_filepath = gb_mask_dir_path + filename
            gb_mask = cv2.imread(gb_mask_filepath, cv2.COLOR_BGR2GRAY)
            gb_mask = gb_mask/255 # rescale to max value of 1
            
            break

    print("DONE.")

    print("Creating fill areas around grain boundary... ", end="")
    # Create fill area maps around gb
    dict_gb_fills = {"0-px": gb_mask}
    np.savez_compressed(gb_mask_dir_path + "gb-fill-0-px.npz", gb_mask)

    px_values = np.arange(1, max_px + 1, 1)

    for px in px_values:
        prev_px_key = "%d-px"%(px-1)
        prev_fill = dict_gb_fills[prev_px_key]
        
        current_fill = np.zeros(prev_fill.shape)
        for r in range(1, current_fill.shape[0]-1):
            for c in range(1, current_fill.shape[1]-1):
                if prev_fill[r,c] == 1:
                    current_fill[r-1:r+2,c-1:c+2] = 1
        
        current_fill_key = "%d-px"%(px)
        dict_gb_fills[current_fill_key] = current_fill
        
        fill_save_path = gb_mask_dir_path + "gb-fill-" + current_fill_key + ".npz"
        np.savez_compressed(fill_save_path, current_fill)

    print("DONE.")

    print("Creating masks at specific distances around grain boundary... ", end="")
    # Create masks around gb
    dict_gb_masks = {"0-px": gb_mask}
    np.savez_compressed(gb_mask_dir_path + "gb-mask-0-px.npz", gb_mask)

    for px in px_values:
        gb_px_mask = dict_gb_fills["%d-px"%(px)]
        for i in np.arange(px-1, -1, -1):
            gb_px_mask -= dict_gb_fills["%d-px"%(i)]
        
        current_mask_key = "%d-px"%(px)
        dict_gb_masks[current_mask_key] = gb_px_mask
        
        mask_save_path = gb_mask_dir_path + "gb-mask-" + current_mask_key + ".npz"
        np.savez_compressed(mask_save_path, gb_px_mask)

    print("DONE.")

    return None



# -----------------------------
def load_gb_fills_and_masks(sample):
    """loads gb fills and masks
    Args:
        sample (str): sample name to be analyzed; dir with same name must be present in sample_data
    Returns:
        dict_gb_fills (dict): dictionary with gb fill maps
        dict_gb_masks (dict): dictionary with gb masks
    """

    sample_dir_path = "sample-data" + SEP + sample + SEP # path to sample directory
    gb_mask_dir_name = "gb_masks"
    gb_mask_dir_path = sample_dir_path + gb_mask_dir_name + SEP

    dict_gb_fills = {}
    dict_gb_masks = {}

    for filename in os.listdir(gb_mask_dir_path):
        if "fill" in filename:
            key = ("%s-%s"%(filename.split("-")[-2], filename.split("-")[-1])).strip(".npz")
            dict_gb_fills[key] = np.load(gb_mask_dir_path + filename)["arr_0"]
        if "mask" in filename:
            key = ("%s-%s"%(filename.split("-")[-2], filename.split("-")[-1])).strip(".npz")
            dict_gb_masks[key] = np.load(gb_mask_dir_path + filename)["arr_0"]

    return (dict_gb_fills, dict_gb_masks)



# -----------------------------
def get_comp_on_gb_masks(sample, dict_gb_masks, dict_px_length, comp_type="at_percent", save=True):
    """gets the compositions on gb masks
    Args:
        sample (str): sample name to be analyzed; dir with same name must be present in sample_data
        dict_gb_fills (dict): dictionary with gb fill maps
        dict_gb_masks (dict): dictionary with gb masks
        comp_type (str, default="at_percent"): composition type: 'at_percent' or 'wt_percent'
        save (bool, default=True): if the dataframe file has to be saved or not
    Returns:
        el_comp_from_gb (pandas dataframe): contains composition of elements as a function of distance from gb
    """

    sample_dir_path = "sample-data" + SEP + sample + SEP # path to sample directory

    dict_EDS_PhaSe, dict_EDS_PhaSe_markerMaps = EDS_PhaSe.load_EDS_PhaSe_cache(sample)
    dict_eds_comp_map = dict_EDS_PhaSe["eds"][comp_type]
    dict_px_length = dict_EDS_PhaSe["info"]["px_length"]

    px_values = np.sort([int(key.split("-")[0]) for key in list(dict_gb_masks.keys())])
    x_distance_values = px_values * dict_px_length["value"]

    eds_el_list = list(dict_eds_comp_map.keys())
    column_names = ["x (px)", "x (micron)"] + eds_el_list
    el_comp_from_gb = pd.DataFrame(columns=column_names)
    el_comp_from_gb["x (px)"] = px_values
    el_comp_from_gb["x (micron)"] = x_distance_values
    el_comp_from_gb = el_comp_from_gb.set_index("x (px)")

    for el in eds_el_list:
        el_eds_comp = dict_eds_comp_map[el]
        el_conc_px_list = []
        
        for px in px_values:
            px_key = "%d-px"%(px)
            gb_px_mask = dict_gb_masks[px_key]
            non_zero_pix = np.count_nonzero(gb_px_mask)
            
            el_comp_on_mask = el_eds_comp * gb_px_mask
            el_comp_total_on_mask = np.sum(el_comp_on_mask)
            el_comp_average_on_mask = el_comp_total_on_mask/non_zero_pix
            
            el_conc_px_list.append(el_comp_average_on_mask)
        
        el_comp_from_gb[el] = np.array(el_conc_px_list)

    if save == True:
        gb_mask_comp_savepath = sample_dir_path + f"{comp_type}-gb_comp_profile.xlsx"
        el_comp_from_gb.to_excel(gb_mask_comp_savepath)

    return (el_comp_from_gb)

