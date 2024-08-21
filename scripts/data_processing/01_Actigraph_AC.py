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
This script obtains valid Activity counts for Actigraph after removing its non-wearing time. 

Input:
- input_folder: Path where study data stores
- watch_acc_path: File where raw acceleration data from smartwatch stores. Each file is acceleration file in one hour.
- Acti_path: File stores where aggregated (calculated from ActiLife software) Activity counts of Actigraph.
- wear_time_folder: Folder path which stores valid wearing time of smartwatch.
- watch_acc_time: File stores smartwatch's valid wearing time for acceleration.
- Acti_no_wear_time: File stores non wearing time of Actigraph. (Created from Actigraph_times.R)

Output:
- output_folder: Folder path which stores ActiAC_file.
- stats_folder: Folder path which stores Acti_miss_file.
- ActiAC_file: Activity counts from Actigraph after removing non-wearing time.
- Acti_miss_file: Miss time statistics about Actigraph.
"""

import pandas as pd
import os
import sys
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from algorithms.ActivityCounts import ActivityCounts
from utils.read_file import load_config
from utils.error_handling import folderErrorHandling

############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
input_root, input_path, watch_folder_path, acti_folder_path, time_path, watch_acc_time, acti_non_wear_time = (
    config.get(key, "") for key in [
        "input_root", "raw_data_folder", "watch_acc_path", "Acti_path", "wear_time_folder", "watch_acc_times_file", "Acti_no_wear_time"
    ]
)

input_folder = os.path.join(input_root, input_path)

# Output and subfolders
output_root, output_path, stats_path, counts_file_path, Acti_miss_file = (
    config.get(key, "") for key in [
        "output_root", "sensor_folder", "stats_folder", "ActiAC_file", "Acti_miss_file"
    ]
)

output_folder = os.path.join(output_root, output_path)
folderErrorHandling(output_folder)

stats_folder = os.path.join(output_root, stats_path)
folderErrorHandling(stats_folder)

time_folder = os.path.join(output_root, time_path)

miss_file_path = os.path.join(stats_folder, Acti_miss_file)

subfolders = [
    f for f in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, f)) and f.isdigit()
]

patient_list = sorted(subfolders, key=lambda x: int(x))

def non_wear_start_end_time(time_folder_p):
    """Obtain each pair of non-wearing start and end time."""
    non_wearing_times= pd.read_csv(os.path.join(time_folder_p, acti_non_wear_time))  
    nw_start_times = pd.to_datetime(non_wearing_times['period_start'])
    nw_end_times = pd.to_datetime(non_wearing_times['period_end'])
    return nw_start_times, nw_end_times

def get_real_ActiAC(df, nw_start_times, nw_end_times):
    """Remove the parts that Actigraph was not wearing."""    
    mask = pd.Series(True, index=df.index)
    for start_time, end_time in zip(nw_start_times, nw_end_times):
        mask &= ~df['time'].between(start_time, end_time)
    filtered_df = df[mask]    
    return filtered_df


def main():
    non_wearing_list = []

    for patient_ID in patient_list:
        input_folder_p = os.path.join(input_folder, patient_ID)
        time_folder_p = os.path.join(time_folder, patient_ID)
        output_folder_p = os.path.join(output_folder, patient_ID)
        folderErrorHandling(output_folder_p)

        watch_times= pd.read_csv(os.path.join(time_folder_p, watch_acc_time))  

        # Obtain whole ActiAC across the study period
        patient_object = ActivityCounts(input_folder_p, patient_ID, watch_times, acti_folder_path = acti_folder_path)
        
        patient_object.get_ActiAC()
        whole_ActiAC = patient_object.ActiAC
        
        # Obtain non-wearing time of Actigraph
        nw_start_times, nw_end_times = non_wear_start_end_time(time_folder_p)
        
        # Obtain real Actigraph counts after removing non-wearing time
        real_ActiAC = get_real_ActiAC(whole_ActiAC, nw_start_times, nw_end_times)
        real_ActiAC.to_csv(os.path.join(output_folder_p, counts_file_path), index=False)

        pct_non_wear = round((len(whole_ActiAC) - len(real_ActiAC)) / len(whole_ActiAC) * 100, 2)
        non_wearing_list.append(pct_non_wear)
        
    non_wearing_df = pd.DataFrame({'ID': patient_list, 'ActiAC-No-Wear [%]': non_wearing_list})
    non_wearing_df.to_csv(miss_file_path, index=False) 

    print(non_wearing_df.describe())

if __name__ == "__main__":
    main()
