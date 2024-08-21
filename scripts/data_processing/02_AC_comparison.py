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
This script calculates Activity counts for smartwatch and compare the counts with Actigraph's Activity counts.

Input:
- input_folder: Path where study data stores
- watch_acc_path: File where raw acceleration data from smartwatch stores. Each file is acceleration file in one hour.
- Acti_path: File stores where aggregated (calculated from ActiLife software) Activity counts of Actigraph.
- wear_time_folder: Folder path which stores valid wearing time of smartwatch.
- sleep_file_name: File stores valid sleep time. (Created from Actigraph_times.R)
- watch_acc_time: File stores smartwatch's valid wearing time for acceleration.

Output:
- output_folder: Folder path which stores counts_file_path.
- fig_folder_path: Folder path which stores figures.
- stats_folder: Folder path which stores Watch_miss_file.
- counts_file_path: Activity counts calculated from Smartwatch acceleration data for every individual.
- AC_compare_file: Comparison metrics of Activity counts of Smartwatch and Actigraph.
- Watch_miss_file: Miss time statistics about the smartwatch.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import csv
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.read_file import check_both_zero, sleep_csv,read_watch_acc_folder, load_config
from algorithms.ActivityCounts import ActivityCounts

############### Define global parameters ###############
# Load configuration and define variables

current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
#input_folder = inputErrorHandling("Enter the data folder: e.g., \n/Users/fanwufan/Documents/PhD/Research/CROCO/StudyData/raw_data/\n")
input_root, input_path, watch_folder_path, acti_folder_path, time_path, sleep_file_name, watch_acc_time = (
    config.get(key, "") for key in [
        "input_root", "raw_data_folder", "watch_acc_path", "Acti_path", "wear_time_folder", "sleep_time_file", "watch_acc_times_file"
    ]
)

input_folder = os.path.join(input_root, input_path)

# Output and subfolders
output_root, output_path, fig_folder_path, stats_path, counts_file_path, AC_compare_file, watch_miss_file = (
    config.get(key, "") for key in [
        "output_root", "sensor_folder", "fig_folder_path", "stats_folder", "WatchAC_file_uncleand", "AC_compare_file", "Watch_miss_file"
    ]
)

time_folder = os.path.join(output_root, time_path)

output_folder = os.path.join(output_root, output_path)
stats_folder = os.path.join(output_root, stats_path)

compare_file_path = os.path.join(output_folder, AC_compare_file)
miss_file_path = os.path.join(stats_folder, watch_miss_file)

# Parameters
str_ID, str_Acti, str_Watch, str_time = (
    config.get(key, "") for key in [
        "str_ID", "str_Acti", "str_Watch", "str_time"
    ]
)

#patient_list = [f"{num:02}" for num in range(1, 3)]

subfolders = [
    f for f in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, f)) and f.isdigit()
]

patient_list = sorted(subfolders, key=lambda x: int(x))

header_row = [str_ID, 'MAE', 'RMSE', 'Mean Difference', 'LoA', 't-statistic', 'p-value [t]', 'Correlation coefficient', 'p-value [corr]']    

charging_list, both_no_wear_list, single_no_wear_list = [], [], []



def main():
    with open(compare_file_path, mode='w', newline='') as csv_file:

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header_row)
        
        for patient_ID in patient_list:
            input_folder_p = os.path.join(input_folder, patient_ID)
            time_folder_p = os.path.join(time_folder, patient_ID)
            output_folder_p = os.path.join(output_folder, patient_ID)
            
            watch_times= pd.read_csv(os.path.join(time_folder_p, watch_acc_time))  

            patient_object = ActivityCounts(input_folder_p, patient_ID, watch_times, time_folder_p, output_folder_p, fig_folder_path, stats_folder, 
                                            watch_folder_path, acti_folder_path, str_Acti, str_Watch, str_time, counts_file_path, sleep_file_name,
                                            charging_list, both_no_wear_list, single_no_wear_list)
            
            patient_object.process()
            
            csv_writer.writerow(patient_object.row_data)
            
    miss_df = pd.DataFrame({str_ID: patient_list, 'Watch-No-Wear [%]': patient_object.charging_list, 'Both-No-Wear [%]': patient_object.both_no_wear_list, 'Single-No-Wear [%]': patient_object.single_no_wear_list})
    miss_df.to_csv(miss_file_path, index=False) 
    
if __name__ == "__main__":
    main()



