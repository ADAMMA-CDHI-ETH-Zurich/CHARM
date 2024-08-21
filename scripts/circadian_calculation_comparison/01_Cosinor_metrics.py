#################################################################################
# Copyright (C) 2024 ETH Zurich
# Circadian Rhythm Metrics based on Activity, body temperature and Heart rate (CHARM)
# Core AI & Digital Biomarker, Acoustic and Inflammatory Biomarkers (ADAMMA)
# Centre for Digital Health Interventions (c4dhi.org)
# 
# Authors: Fan Wu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#         http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################


"""
Description:
This script calculates cosinor metrics (i.e., amplitude, mesor, acrophase) from the sensor data.

Input:
- input_folder: Path where sensor data stores
- ActiAC_file: File where activity counts of Actigraph stores.
- WatchAC_file: File where activity counts of the smartwatch stores.
- HR_file: File where heart rate obtained from the smartwatch stores.
- HRV_file: File where heart rate variablity stores.
- CBT_file: File where core and skin temperature stores.

Output:
- output_folder: Folder path which stores calculated cosinor metrics.
- fig_folder_path: Folder path which stores figures.
- subfolder: Folder path which stores cosinor metrics for each individual.
- output_csv: File path which stores cosinor metrics for all individuals.
"""


from CosinorPy import cosinor, cosinor1, cosinor_nonlin
import importlib
importlib.reload(cosinor)
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.error_handling import folderErrorHandling
from utils.read_file import load_config


############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
root, input_path, ActiAC_file, WatchAC_file, HR_file, HRV_file, CBT_file = (
    config.get(key, "") for key in [
        "output_root", "sensor_folder", "ActiAC_file", "AC_file",  "HR_file", "HRV_file", "CBT_file"
    ]
)

input_folder = os.path.join(root, input_path)

# Output folder and files
fig_folder_path, output_path, subfolder, output_csv = (
    config.get(key, "") for key in [
        "fig_folder_path", "CR_folder", "CR_model_folder", "CR_model_file"
    ]
)

output_folder = os.path.join(root, output_path)

folderErrorHandling(output_folder)
fig_folder = os.path.join(output_folder, fig_folder_path)
folderErrorHandling(fig_folder)
output_subfolder = os.path.join(output_folder,subfolder)
folderErrorHandling(output_subfolder)

# Strings
str_time, str_Acti, str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4 = (
    config.get(key, "") for key in [
        "str_time", "str_Acti", "str_Watch",  "str_CBT", "str_SkinT", "str_HR", "str_HRV1", "str_HRV2", "str_HRV3", "str_HRV4"
    ]
)

str_to_test = [str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4]

subfolders = [
    f for f in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, f)) and f.isdigit()
]

patient_list = sorted(subfolders, key=lambda x: int(x))


time_interval = 10 # Aggerate to 10T



def read_csv_time(patient_ID, file):
    """Read file and transfer time into datatime format."""
    df = pd.read_csv(os.path.join(input_folder, patient_ID, file))
    df[str_time] = pd.to_datetime(df[str_time])
    return df



def read_sensor_data(patient_ID): 
    """Read sensor data and aggregate them into 10-minute interval."""
    df_ActiAC = read_csv_time(patient_ID, ActiAC_file)
    df_ActiAC = df_ActiAC.rename(columns={'AC': str_Acti})
    df_ActiAC = df_ActiAC[[str_time, str_Acti]]

    df_WatchAC = read_csv_time(patient_ID, WatchAC_file)
    df_WatchAC = df_WatchAC[[str_time, str_Watch]]

    df_CBT_SkinT = read_csv_time(patient_ID, CBT_file)
    df_CBT = df_CBT_SkinT[[str_time, str_CBT]]
    df_SkinT = df_CBT_SkinT[[str_time, str_SkinT]]

    df_HR = read_csv_time(patient_ID, HR_file)
    df_HR = df_HR[[str_time, str_HR]]

    df_HRV = read_csv_time(patient_ID, HRV_file)
    names_HRV = [str_HRV1, str_HRV2, str_HRV3, str_HRV4]
    df_HRVs = [df_HRV[[str_time, name_HRV]] for name_HRV in names_HRV]
    return [df_ActiAC, df_WatchAC, df_CBT, df_SkinT, df_HR] + df_HRVs


def aggregate_df(list_df):
    """For ActiAC and WatchAC, aggregate based on sum, for CBT, SKinT and HR, aggregated based on mean"""
    agg_list = list_df.copy() 
    agg_list[:2] = [aggregate_sum(elem) for elem in agg_list[:2]]
    agg_list[2:5] = [aggregate_mean(elem) for elem in agg_list[2:5]]
    return agg_list
    

def aggregate_mean(df):
    """Aggregate based on mean."""
    df.set_index(str_time, inplace=True)
    df_downsampled = df.resample(str(time_interval) + 'T').mean()
    df_downsampled.reset_index(inplace=True)  
    return df_downsampled

def aggregate_sum(df):
    """Aggregate based on sum"""
    df.set_index(str_time, inplace=True)
    df_downsampled = df.resample(str(time_interval) + 'T').sum()
    df_downsampled.reset_index(inplace=True) 
    return df_downsampled

def to_cosinor_format(df, label):
    """Change to the dataformat applied in cosinor model, x is minute index, y is measurement, test is label."""
    df2 = pd.DataFrame()
    df2['x'] = (df['time'].dt.hour * 60 + df['time'].dt.minute)/10
    df2['y'] = df.iloc[:, -1]
    df2['test'] = label
    return df2

def scale_measure(df, scaler):
    """Scale the measurement"""
    scaled_df = df.copy() 
    for column in df.columns:
        if column == 'y': 
            scaled_column = scaler.fit_transform(df[column].values.reshape(-1, 1))
            scaled_df[column] = pd.Series(scaled_column.flatten(), index=df.index)
    return scaled_df

def cosinor_metrics(df, time_period, pairs, patient_ID=None, model_csv=None, save_folder=None, plot_figure=None, compare_csv=None):
    """Call cosinor functions and calculate cosinor metrics."""
    # Identify the best models and/or the best periods (possible periods can be given as an interval or as a single value).
    df_results = cosinor.fit_group(df, n_components = [1], period=time_period, plot=False, plot_phase=False) #folder=""

    # Get the best fitting periods with criterium 'RSS' (```reverse=False``` means lower is better)
    df_best_fits = cosinor.get_best_fits(df_results, n_components = [1], criterium='RSS', reverse = False)
    
    if patient_ID is not None:
        df_best_fits.insert(loc=0, column='ID', value=patient_ID)
    
    df_best_fits['acrophase'] = df_best_fits['acrophase'].apply(acro_neg_to_pos)
    df_best_fits['time'] = df_best_fits['acrophase'].apply(acro_to_time)
    df_best_fits['hour'] = df_best_fits['acrophase'].apply(acro_to_hour)

    if model_csv is not None:
        df_best_fits.to_csv(model_csv, index=False)
    
    if plot_figure is not None:
    # plot these models, by default the criterium is p-value)
        df_best_models = cosinor.get_best_models(df, df_results, n_components = [1])    
        cosinor.plot_df_models(df, df_best_models, folder=save_folder)
    
    if compare_csv is not None:
        # Comparison, only 1-component model
        compare_cosinor1 = cosinor1.test_cosinor_pairs(df, pairs, period=time_period, folder=plot_figure)
        compare_cosinor1.to_csv(compare_csv, index=False)

    '''
    # Only 1-component model can be used, but the statistics is much richer...
    df_results = cosinor1.fit_group(df, period=[time_period])
    df_results.insert(loc=0, column='ID', value=patient_ID)
    df_results.to_csv(os.path.join(output_folder,patient_ID+"_cosinor1.csv"), index=False)

    '''

def acro_neg_to_pos(value):
    """Change the negative acrophase value to positive by adding 2*pi"""
    two_pi = 2 * np.pi
    while value < 0:
        value += two_pi
    return value

def acro_to_hour(acrophase):
    """Change the acrophase value to hour (Hour: Minute)"""
    time = acro_to_time(acrophase)
    hours = int(time)
    minutes = int((time % 1) * 60)
    return f'{hours:02d}:{minutes:02d}'

def acro_to_time(acrophase):
    """Change the acrophase value to time value"""
    time = 24 - 24 * acrophase / (2 * np.pi)
    return time


def main():
    for patient_ID in patient_list:
        print("="*30)
        print(patient_ID)
        
        sensor_data = read_sensor_data(patient_ID)
    
        sensor_data_agg = aggregate_df(sensor_data)
        
        cr_data = [to_cosinor_format(df, df.columns[-1]) for df in sensor_data_agg]
    
        scaler = MinMaxScaler()
    
        cr_data_scaled = [scale_measure(df, scaler) for df in cr_data]
    
        cr_data_mergerd = pd.concat(cr_data_scaled, ignore_index=True)
        cr_data_mergerd = cr_data_mergerd.dropna(subset=['y'])
    
        #pairs = (["AC(Actigraph)", "CBT"], )
        pairs = tuple([[str_Acti, item] for item in str_to_test])
        
        cosinor_metrics(df = cr_data_mergerd, time_period = int(24*60/time_interval), pairs=pairs, patient_ID = patient_ID, model_csv=os.path.join(output_subfolder, patient_ID+"_models.csv"))
        print("="*30)
        
    #output_strings = ['models', 'cosinor1', 'compare1', 'CI', 'nonlinear', 'bootstrap', 'comparelm1']#, '3comp', 'comparelmbest', 'comparebootstrap']
    output_strings = ['models']
    
    for substr in output_strings:
        merged_df = pd.DataFrame()
        
        filtered_files = [filename for filename in os.listdir(output_subfolder) if substr in filename and filename.endswith('.csv')]
        sorted_files = sorted(filtered_files, key=lambda x: int(x.split('_')[0]))
        
        for filename in sorted_files:
            df = pd.read_csv(os.path.join(output_subfolder, filename))
            merged_df = pd.concat([merged_df, df])
        merged_df.to_csv(os.path.join(output_folder, output_csv), index=False)
    

    
if __name__ == "__main__":
    main()


