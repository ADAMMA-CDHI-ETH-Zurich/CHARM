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
This script obtains smartwatch valid wearing time for acceleration and heart rate. It excludes the time when smartwatch was charging or when no related file was not recorded.

Input:
- input_folder: Path where study data stores
- process_folder: Path where charging time stores
- battery_file_name: File stores the time fragments after removing smartwatch charging time. 
- acc_folder: Subfolder path where acceleration data of the smartwatch stores. Inside the folder, each file contains 3-axis acceleration in one hour.
- hr_folder: Subfolder path where heart rate data of the smartwatch stores. Inside the folder, each file contains heart rate data in one hour.

Output:
- wear_time_folder: Folder path which stores valid wearing time of smartwatch.
- acc_times_file: File stores valid wearing time for acceleration.
- hr_times_file: File stores valid wearing time for heart rate.
"""

import os
import datetime
import pandas as pd
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
input_root, input_path, battery_file_name, acc_folder, hr_folder = (
    config.get(key, "") for key in [
       "input_root", "raw_data_folder", "battery_times_file", "watch_acc_path", "hr_folder_path"
    ]
)

input_folder = os.path.join(input_root, input_path)

# Output and subfolders
output_root, output_path, acc_times_file, hr_times_file = (
    config.get(key, "") for key in [
        "output_root", "wear_time_folder","watch_acc_times_file", "hr_times_file"
    ]
)
output_folder = os.path.join(output_root, output_path)

subfolders = [
    f for f in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, f)) and f.isdigit()
]

patient_list = sorted(subfolders, key=lambda x: int(x))


def files_miss(start_time, end_time, folder):
    """Find the specific hours when file was not recorded within the whole 2-week study period"""
    expected_files = []
    current_time = start_time
    while current_time < end_time:
        file_name = current_time.strftime('%d.%m.%y_%H') + '.csv'
        expected_files.append(file_name)
        current_time += datetime.timedelta(hours=1)

    # Get the list of files in the folder
    files = os.listdir(folder)
    missing_files = set(expected_files) - set(files)
    missing_files = sorted(list(missing_files), key=lambda x: datetime.datetime.strptime(x[:-4], "%d.%m.%y_%H"))
    return missing_files

def read_times(time_file):
    """Read each pair of start and end wearing time"""
    times= pd.read_csv(time_file)  
    times['Start'] = pd.to_datetime(times['Start'])
    times['End'] = pd.to_datetime(times['End'])
    
    start_times = pd.to_datetime(times['Start'])
    end_times = pd.to_datetime(times['End'])
    return start_times, end_times

def files_to_time(missing_files):
    """Identifies consecutive files with hour sequences and merges them into time intervals where no files were found."""
    nofile_starts = []
    nofile_ends = []
    current_start = None
    previous_datetime = None
    
    for file in missing_files:
        datetime_str = file[:-4]  # Remove the ".csv" extension
        current_datetime = datetime.datetime.strptime(datetime_str, "%d.%m.%y_%H")
        if previous_datetime and current_datetime - previous_datetime > datetime.timedelta(hours=1):
            if current_start:
                if current_start == previous_datetime.strftime("%d.%m.%y_%H"):
                    tmp = datetime.datetime.strptime(current_start, "%d.%m.%y_%H")
                    nofile_starts.append(tmp)
                    nofile_ends.append(tmp+pd.Timedelta(hours=1))
                else:
                    nofile_starts.append(datetime.datetime.strptime(current_start, "%d.%m.%y_%H"))
                    nofile_ends.append(previous_datetime+pd.Timedelta(hours=1))
            current_start = None
    
        if not current_start:
            current_start = current_datetime.strftime('%d.%m.%y_%H')
    
        previous_datetime = current_datetime
    
    if current_start:
        if current_start == previous_datetime.strftime("%d.%m.%y_%H"):
            tmp = datetime.datetime.strptime(current_start, "%d.%m.%y_%H")
            nofile_starts.append(tmp)
            nofile_ends.append(tmp+pd.Timedelta(hours=1))
            
        else:
            nofile_starts.append(datetime.datetime.strptime(current_start, "%d.%m.%y_%H"))
            nofile_ends.append(previous_datetime+pd.Timedelta(hours=1))
    return nofile_starts, nofile_ends



def combine_battery_nofile(start_times, end_times, nofile_starts, nofile_ends):
    """combine and merge the times with battery miss and times when no file was recorded."""
    combined_starts = pd.concat([start_times, pd.Series(nofile_ends)])
    combined_starts = combined_starts.sort_values()
    combined_starts = combined_starts.reset_index(drop=True)
    
    combined_ends = pd.concat([end_times, pd.Series(nofile_starts)])
    combined_ends = combined_ends.sort_values()
    combined_ends = combined_ends.reset_index(drop=True)
    
    problem_indices = []
        
    # Check if each item in endtime is later than the corresponding starttime
    for i, _ in enumerate(combined_starts):
        if not combined_starts.iloc[i] < combined_ends.iloc[i]:
            #print("Battery charging time has problem", i, "*start*", combined_starts.iloc[i], "*end*", combined_ends.iloc[i])
            problem_indices.append(i)
    
    combined_starts.drop(problem_indices, inplace=True)
    combined_ends.drop(problem_indices, inplace=True)
    
    combined_times = pd.DataFrame({'Start': combined_starts, 'End': combined_ends})
    combined_times = combined_times.reset_index(drop=True)
    return combined_times
    

def main():
    for patient_ID in patient_list:

        input_folder_p = os.path.join(input_folder, patient_ID)    
        output_folder_p = os.path.join(output_folder, patient_ID)
        folderErrorHandling(output_folder_p)
    
        start_times, end_times = read_times(os.path.join(output_folder_p, battery_file_name))
        
        for subfolder_path, output_str in zip([acc_folder, hr_folder], [acc_times_file, hr_times_file]):
        
            missing_files = files_miss(start_times.iloc[0], end_times.iloc[-1], os.path.join(input_folder_p, subfolder_path))
            
            nofile_starts, nofile_ends = files_to_time(missing_files)
            
            combined_times = combine_battery_nofile(start_times, end_times, nofile_starts, nofile_ends)
        
            combined_times.to_csv(os.path.join(output_folder_p, output_str), index=False)
            


if __name__ == "__main__":
    main()
