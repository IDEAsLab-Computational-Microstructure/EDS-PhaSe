"""
contains functions to assist eds functions
List of functions:
    convert_comp_at_to_wt: convert alloy composition from atomic percent to weight percent
    convert_comp_wt_to_at: convert alloy composition from weight percent to atomic percent
    max_intensity_channel: extract channel with highest intensity 
    max_variance_channel: extract channel with highest variance 
    RGB2GRAY_img: color to grayscale conversion using (openCV) cv2 'COLOR_RGB2GRAY' method
"""

import numpy as np
import json
import os

SEP = os.sep


# -----------------------------------
def convert_comp_at_to_wt(dict_comp_at):
    
    """
    function to convert alloy composition from at_percent to wt_percent
    INPUT:
        dict_comp_at: (dict) input alloy composition in at_percent
    OUTPUT:
        dict_comp_wt: (dict) alloy composition in wt_percent
    """
    
    db_el_path = "_program-data" + SEP + "database-element.json"

    with open(db_el_path, "r") as f:
        dict_db_el = json.load(f)

    el_list = list(dict_comp_at.keys())
    el_wt_list = [dict_comp_at[el] * dict_db_el[el]["atomic-weight"] for el in el_list]
    el_wt_percent_list = [el_wt * 100 / np.sum(el_wt_list) for el_wt in el_wt_list]

    dict_comp_wt = dict(zip(el_list, el_wt_percent_list))            

    return (dict_comp_wt)



# -----------------------------------
def convert_comp_wt_to_at(dict_comp_wt):

    """
    function to convert alloy composition from wt_percent to at_percent
    INPUT:
        dict_comp_wt: (dict) input alloy composition in wt_percent
    OUTPUT:
        dict_comp_at: (dict) alloy composition in at_percent
    """
    
    db_el_path = "_program-data" + SEP + "database-element.json"

    with open(db_el_path, "r") as f:
        dict_db_el = json.load(f)

    el_list = list(dict_comp_wt.keys())
    el_mole_list = [dict_comp_wt[el] / dict_db_el[el]["atomic-weight"] for el in el_list]
    el_at_percent_list = [el_mole * 100 / np.sum(el_mole_list) for el_mole in el_mole_list]

    dict_comp_at = dict(zip(el_list, el_at_percent_list))            

    return (dict_comp_at)
    
    
    
# -----------------------------
def max_intensity_channel(img):
    
    """
    function to extract the highest intensity channel from a multiple channel image
    INPUT:
        img: (numpy array) input image
    OUTPUT:
        max_int_channel_img: (numpy array) single-channel image from channel with highest intensity
    """
    
    if img.shape[2] > 1:
        ch_max_int_list = [np.max(img[:,:,i]) for i in range(0, img.shape[2])] # max intesity in each channel
        ch_to_use = np.argmax(ch_max_int_list) # index of channel with max intensity
        max_int_channel_img = img[:, :, ch_to_use]
        max_int_channel_img = np.array(max_int_channel_img, dtype=float)
        
        return (max_int_channel_img)
    
    else:
        return (img)



# -----------------------------
def max_variance_channel(img):
    
    """
    function to extract the highest intensity channel from a 3-channel color image
    INPUT:
        img: (numpy array) input image
    OUTPUT:
        max_var_channel_img: single-channel image from channel with highest variance
    """
    
    if img.shape[2] > 1:
        ch_var_list = [np.var(img[:,:,i]) for i in range(0, img.shape[2])] # variance in each channel
        ch_to_use = np.argmax(ch_var_list) # index of channel with max variance
        max_var_channel_img = img[:, :, ch_to_use]
        max_var_channel_img = np.array(max_var_channel_img, dtype=float)

        return (max_var_channel_img)
    
    else:
        return (img)



# -----------------------------
def RGB2GRAY_img(img):
    
    """
    function to convert a color image to single-channel grayscale image using cv2 'COLOR_RGB2GRAY' method
    INPUT:
        img: (numpy array) input image
    OUTPUT:
        img_grayscale: (numpy array) grayscale single channel image
    """
    
    import cv2

    if img.shape[2] > 1:
        img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_grayscale = np.array(img_grayscale, dtype=float)

        return (img_grayscale)

    else:
        return (img)