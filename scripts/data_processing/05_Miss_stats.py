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
import matplotlib.pyplot as plt
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.read_file import load_config
from utils.error_handling import folderErrorHandling


############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
output_root, stats_path, a_miss_file, w_miss_file, c_miss_file  = (
    config.get(key, "") for key in [
        "output_root","stats_folder", "Acti_miss_file", "Watch_miss_file", "CBT_miss_file"
    ]
)

stats_folder = os.path.join(output_root, stats_path)
folderErrorHandling(stats_folder)

acti_miss_file = os.path.join(stats_folder, a_miss_file) 
watch_miss_file = os.path.join(stats_folder, w_miss_file) 
core_miss_file = os.path.join(stats_folder, c_miss_file)

# Output and subfolders
all_miss_path, = (
    config.get(key, "") for key in [
        "Overall_miss_file",
    ]
)

all_miss_file = os.path.join(stats_folder, all_miss_path)


def merge_miss():
    """Merge three types of miss files."""
    # Read Acti miss file
    acti_miss_df = pd.read_csv(acti_miss_file)  
    
    # Read Watch miss file
    watch_df = pd.read_csv(watch_miss_file)  
    watch_miss_df = watch_df[['ID', 'Watch-No-Wear [%]']]
    
    # Check if the CORE miss file exists
    core_miss_df = pd.read_csv(core_miss_file)  
    
    miss_df = pd.merge(pd.merge(acti_miss_df, watch_miss_df, on='ID'), core_miss_df, on='ID')
    
    miss_df = miss_df.drop(columns='ID')

    column_names = ['Actigraph', 'Smartwatch', 'CORE']
    miss_df.columns = column_names
    
    miss_df.to_csv(all_miss_file, index=False)  
    
    return miss_df

def read_or_calculate_miss():
    """Decide if read the existing miss file or recalculate the file."""
    if os.path.exists(core_miss_file):
        regenerate = input("The overall miss file already exists. Do you want to regenerate it? (Y/N): ")
        if regenerate == 'Y':
            miss_df = merge_miss()  
        elif regenerate == 'N':
            miss_df = pd.read_csv(all_miss_file)  
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")
    else:
        miss_df = merge_miss()  
    return miss_df
    


def main():
    miss_df = read_or_calculate_miss()

    print(miss_df.describe())

if __name__ == "__main__":
    main()
