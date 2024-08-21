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
This script cleans heart rate (HR) data, calculates four heart rate variability (HRV) metrics and visulize it with acceleration data.

Input:
- input_folder: Path where study data stores
- hr_folder_path: File where HR data from smartwatch stores. Each file is acceleration file in one hour.
- hr_times_file: File stores smartwatch's valid wearing time for HR.
- AC_file: File stores where cleaned (after removing non-wearing) Activity counts of Actigraph.

Output:
- data_folder: Actually also input folder. Folder path which stores Actigraph's counts and should store HR and HRV files.
- fig_folder_path: Folder path which stores figures.
- hr_file_name: Path which stores HR data for every individual.
- hrv_file_name: Path which stores HRV data for every individual.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.read_file import read_hr_folder, load_config
from utils.visualization import scatter
from utils.error_handling import folderErrorHandling


############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
input_root, input_path, hr_folder_path, time_file_name, ac_file_name  = (
    config.get(key, "") for key in [
        "input_root", "raw_data_folder", "hr_folder_path", "hr_times_file", "AC_file"
    ]
)

input_folder = os.path.join(input_root, input_path)

# Output and subfolders
output_root, output_path, fig_folder_path, hr_file_name, hrv_file_name = (
    config.get(key, "") for key in [
        "output_root", "sensor_folder", "fig_folder_path", "HR_file", "HRV_file"
    ]
)

data_folder = os.path.join(output_root, output_path)

# Parameters
str_time, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4, str_acc = (
    config.get(key, "") for key in [
        "str_time", "str_HR", "str_HRV1", "str_HRV2", "str_HRV3", "str_HRV4", "str_Acti"
    ]
)

subfolders = [
    f for f in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, f)) and f.isdigit()
]

patient_list = sorted(subfolders, key=lambda x: int(x))


def calculate_hrv(data):
    """Calculate 4 HRV metrics from RR interval."""
    if len(data) > 5:
        nn_intervals = data['hrIbi']
        # Calculate Mean RR interval: Mean of the SD of all RR intervalsin the segment
        mean_rr = np.mean(nn_intervals)
        # Calculate SDNN:  Standard deviation of all RR intervals
        sdnn = np.std(nn_intervals)
        # Calculate RMSSD (Root Mean Square of Successive Differences): Square root of the mean of the sum of squares of differences between adjacent RR intervals
        rmssd = np.sqrt(np.mean(np.diff(nn_intervals) ** 2))
        # Calculate pNN50 (Percentage of NN50 Intervals): Percentage of differences between adjacent RR intervals that are greater than 50 msec
        nn50_count = sum(np.abs(np.diff(nn_intervals)) > 50)
        total_intervals = len(nn_intervals)
        pnn50 = (nn50_count / total_intervals) * 100
        return [mean_rr, sdnn, rmssd, pnn50]
    else:
        return None


def main():
    for patient_ID in patient_list:        
        input_folder_p = os.path.join(input_folder, patient_ID)
        data_folder_p = os.path.join(data_folder, patient_ID)

        df_hr = read_hr_folder(os.path.join(input_folder_p, hr_folder_path))
        df_hr.to_csv(os.path.join(data_folder_p, hr_file_name), index=False)
            
        # delete data when heart rate is smaller than 30 or over 240, ibi is  time between heartbeats is 1000 ms - 60 beats/min
        ibi = df_hr[(df_hr['hrIbi'] >= 25) & (df_hr['hrIbi'] <= 2000)]
    
        # Calculate HRV metrics in 10-minute interval
        windowed_data = ibi.groupby(pd.Grouper(key=str_time, freq='10Min'))
        df_hrv = pd.DataFrame(columns=[str_time, str_HRV1, str_HRV2, str_HRV3, str_HRV4])
        for window_start, group in windowed_data:
            hrv = calculate_hrv(group)        
            if hrv is not None:
                data = {str_time: [window_start], str_HRV1: [hrv[0]], str_HRV2: [hrv[1]], str_HRV3: [hrv[2]], str_HRV4: [hrv[3]]}
                hrv_data = pd.DataFrame(data)            
                df_hrv = pd.concat([df_hrv, hrv_data], ignore_index=True)
                
        df_hrv.to_csv(os.path.join(data_folder_p, hrv_file_name), index=False)
        
        print("Compare heart rate and acceleration ....\n")
        # Correlation between ActiAC and heart rate
        df_acc = pd.read_csv(os.path.join(data_folder_p, ac_file_name))
        df_acc[str_time] = pd.to_datetime(df_acc[str_time])
                
        merged_df = pd.merge(df_acc, df_hr, on=str_time, how='inner')
        
        data = merged_df[[str_time, str_acc, str_HR]]
        data.dropna(inplace=True)    
        data = data.reset_index(drop=True)
        
        x = data[str_time]
        scaler = MinMaxScaler()
        y1 = scaler.fit_transform(data[[str_acc]])
        y2 = scaler.fit_transform(data[[str_HR]])
        
        fig_folder_p = os.path.join(data_folder, patient_ID, fig_folder_path)
        folderErrorHandling(fig_folder_p)

        with plt.style.context('seaborn'):
            scatter(x, y1, y2, str_acc, str_HR, plot_save=True, fig_folder=fig_folder_p)
        
        correlation = data[str_acc].corr(data[str_HR])

    
if __name__ == "__main__":
    main()




