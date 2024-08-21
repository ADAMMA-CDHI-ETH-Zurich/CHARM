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
This script read core body temperature file, remove low quality data and calculate the missing time.

Input:
- input_folder: Path where study data stores
- study_period_file: Path where study and end time of each participant stores, e,g., 2023-08-01 10:00, 2023-08-14 12:00
- input_core_file: File stores CBT data of each individual, downloaded from CORE Cloud.

Output:
- data_folder: Folder path which stores cleaned CBT data.
- fig_folder_path: Folder path which stores figures.
- core_file_name: Path which stores CBT data for every individual.
- stats_folder: Path which stores CBT_miss_file.
- CBT_miss_file: Miss time statistics about CBT.
"""

import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.read_file import core_csv, load_config
from utils.visualization import scatter

############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
input_root, input_path, time_file_path, input_core_file  = (
    config.get(key, "") for key in [
        "input_root", "raw_data_folder", "study_period_file", "raw_core_file"
    ]
)

input_folder = os.path.join(input_root, input_path)
time_file = os.path.join(input_root, time_file_path)

# Output and subfolders
output_root, output_path, fig_folder_path, core_file_name, stats_path, CBT_miss_file = (
    config.get(key, "") for key in [
        "output_root", "sensor_folder", "fig_folder_path", "CBT_file", "stats_folder", "CBT_miss_file"
    ]
)

data_folder = os.path.join(output_root, output_path)
stats_folder = os.path.join(output_root, stats_path)

core_miss_file = os.path.join(stats_folder, CBT_miss_file)

# Parameters
str_ID, str_time, str_CBT, str_SkinT = (
    config.get(key, "") for key in [
        "str_ID", "str_time", "str_CBT", "str_SkinT"
    ]
)

subfolders = [
    f for f in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, f)) and f.isdigit()
]

patient_list = sorted(subfolders, key=lambda x: int(x))

#patient_list = [f"{num:02}" for num in range(1, 39) if num not in [30, 36]]


def core_miss_pct(df_core, start_time, end_time):
    """Calculate CORE miss time."""
    expected_range = pd.date_range(start=start_time, end=end_time, freq='1min')
    existing_df = df_core[str_time]   
    existing_df = existing_df.drop_duplicates(keep='last')

    expected_df = pd.DataFrame({'DateTime': expected_range})        
    expected_df = expected_df.drop_duplicates(keep='last')
    
    pct_non_wear = round((len(expected_df) - len(existing_df)) / len(expected_df) * 100, 2)
    return pct_non_wear
            

def main():
    times= pd.read_csv(time_file)  
    no_wear_list = []

    for patient_ID in patient_list:    
        print("*"*20)
        print(patient_ID)
        input_file = os.path.join(input_folder, patient_ID, input_core_file)
        data_folder_p = os.path.join(data_folder, patient_ID)

        # Read core time, remove invalid data and save
        df_core, start_time, end_time = core_csv(input_file, times, patient_ID)
        df_core.to_csv(os.path.join(data_folder_p, core_file_name), index=False)
        
        # Visualize CBT and SkinT
        fig_folder_p = os.path.join(data_folder, patient_ID, fig_folder_path)
        with plt.style.context('seaborn'):
            scatter(df_core[str_time], df_core[str_CBT], df_core[str_SkinT], str_CBT, str_SkinT, plot_save=True, fig_folder=fig_folder_p,  ylim_low = "min")
            
        # Calculate the miss time
        pct_non_wear = core_miss_pct(df_core, start_time, end_time)
        no_wear_list.append(pct_non_wear)
    
    # Save miss time
    core_miss_df = pd.DataFrame({str_ID: patient_list, 'Core-No-Wear [%]': no_wear_list})
    core_miss_df.to_csv(core_miss_file, index=False) 


if __name__ == "__main__":
    main()
