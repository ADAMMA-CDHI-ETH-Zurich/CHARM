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
This script obtains time fragments after excluding smartwatch charging time. 

Input:
- input_folder: Path where study data stores
- watch_folder_path: Subfolder path where watch data stores
- battery_folder: Subfolder path where battery data of the smartwatch stores. Inside the folder, each battery file contains battery level in one hour.
- time_file: Path where study and end time of each participant stores, e,g., 2023-08-01 10:00, 2023-08-14 12:00
- patient_ID: Patient ID

Output:
- battery_miss.csv: file stores the time fragments after removing smartwatch charging time. 
"""

import pandas as pd
import datetime
from tqdm import tqdm
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from matplotlib import pyplot as plt
from utils.read_file import battery_csv, load_config
from utils.error_handling import folderErrorHandling


############### Define global parameters ###############
# Load configuration and define variables

current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

patient_ID = "01" 

# Input folder and files
input_root, input_path, watch_folder_path, time_file_path = (
    config.get(key, "") for key in [
        "input_root", "raw_data_folder", "battery_folder_path", "study_period_file"
    ]
)

input_folder = os.path.join(input_root, input_path)
time_file = os.path.join(input_root, time_file_path)

#input_folder = inputErrorHandling("Enter the data folder: e.g., \n/Users/fanwufan/Documents/PhD/Research/CROCO/StudyData/raw_data/\n")
battery_folder = os.path.join(input_folder, patient_ID, watch_folder_path)

# Output folder and files
output_root, output_path, battery_times_file = (
    config.get(key, "") for key in [
        "output_root", "wear_time_folder", "battery_times_file"
    ]
)

output_folder = os.path.join(output_root, output_path, patient_ID)

# Output folder and files
folderErrorHandling(output_folder)
main_output = os.path.join(output_folder, battery_times_file)

# Parameter to detect charging time
window_size = 40 # 4 for old version, rate is 5-6 minutes # 40 for newer version, rate is 10 sec
half_window = int(1/2*window_size)
buffer_minute = 5
increase_thres = 5 # 5 for old version

def read_times(file_name, patient_ID):
    """Read start and end time in the study period."""
    times= pd.read_csv(file_name)
    start_str = times.loc[times['ID'] == int(patient_ID), 'Start'].iloc[0]
    end_str = times.loc[times['ID'] == int(patient_ID), 'End'].iloc[0]
    start_time = datetime.datetime.strptime(start_str, "%d.%m.%y %H:%M")
    end_time = datetime.datetime.strptime(end_str, "%d.%m.%y %H:%M")
    return start_time, end_time

def find_charging_time(battery_folder, start_time, end_time): 
    """Find charging time fragments based on charging level and status."""
    file_names = os.listdir(battery_folder)
    csv_names = [file_name for file_name in file_names if file_name.endswith('.csv')]
    sorted_csv_names = sorted(csv_names, key=lambda x: datetime.datetime.strptime(x.rsplit('.', 1)[0], "%d.%m.%y_%H"), reverse=False)
    
    df_battery = pd.DataFrame()
    for file in tqdm(sorted_csv_names):
        timestamp = datetime.datetime.strptime(file, "%d.%m.%y_%H.csv")
        if start_time <= timestamp < end_time:
            file_path = os.path.join(battery_folder, file)
            df = battery_csv(file_path)
            if df is not None:
                df_battery = pd.concat([df_battery, df])

    increase_mask = (df_battery['state'].shift(1).isin([1,2])) & (df_battery['state'].isin([3, 4, 5, 6]))
    decrease_mask = (df_battery['state'].shift(-1).isin([1,2])) & (df_battery['state'].isin([3, 4, 5, 6]))

    increase_datetime = df_battery.loc[increase_mask, 'time']
    charge_starttime = increase_datetime - datetime.timedelta(minutes=2)

    decrease_datetime = df_battery.loc[decrease_mask, 'time']
    charge_endtime = decrease_datetime + datetime.timedelta(minutes=2)
    
    charge_starttime = charge_starttime.reset_index(drop=True)
    charge_endtime = charge_endtime.reset_index(drop=True)
    
    if len(charge_starttime) < len(charge_endtime):
        del charge_endtime[0]
        charge_endtime = charge_endtime.reset_index(drop=True)
    
    elif len(charge_starttime) > len(charge_endtime):
        charge_starttime = charge_starttime[:-1]

    # Check if each item in endtime is later than the corresponding starttime
    for i in range(len(charge_starttime)):
        if charge_endtime[i] < charge_starttime[i]:
            print("Battery charging time has problem")

    return df_battery, charge_starttime, charge_endtime


def charging_timev1(battery_folder, start_time, end_time):
    """First method to look for charging time based on charging level and status."""
    df_battery, charge_starttime, charge_endtime = find_charging_time(battery_folder, start_time, end_time)
    start_times = pd.concat([pd.Series([start_time]), charge_endtime])
    end_times = pd.concat([charge_starttime, pd.Series([end_time])])
    start_times = start_times.reset_index(drop=True)
    end_times = end_times.reset_index(drop=True)

    return df_battery, start_times, end_times


def charging_timev2(df_battery, start_time, end_time):
    """Second method to look for charging time based on differences between charging levels in the moving window."""
    df_battery.reset_index(drop=True, inplace=True)
    df_battery['time'] = pd.to_datetime(df_battery['time'])
    
    # Check if the 'level' values in the window first decrease and then increase
    charge_start_time = None  
    end_times2 = []
    i = 0
    while i < len(df_battery) - window_size:
        current_window = df_battery.iloc[i : i + window_size]  
        level_diff = current_window['level'].diff()  # Calculate the differences between consecutive 'level' values
        if level_diff[:half_window].all() <= 0 and level_diff[half_window:].sum() > increase_thres:
            charge_start_time = current_window['time'].iloc[0]
            end_times2.append(charge_start_time - datetime.timedelta(minutes=buffer_minute))
            i = i + window_size
        else:
            i = i + 1
    end_times2.append(end_time) 
    
    # Check if the 'level' values in the window first increase and then decrease
    charge_end_time = None  
    start_times2 = []
    start_times2.append(start_time)
    i = 0
    while i < len(df_battery) - window_size:
        current_window = df_battery.iloc[i : i + window_size] 
        level_diff = current_window['level'].diff()  # Calculate the differences between consecutive 'level' values
        statuses = current_window['state']
        if level_diff[:half_window].sum() > 5 and level_diff[half_window:].sum() <= -1:
            charge_end_time = current_window['time'].iloc[-1]
            start_times2.append(charge_end_time + datetime.timedelta(minutes=buffer_minute))
            i = i + window_size
        elif (statuses[:half_window] > 2).all() & (statuses[half_window+1:window_size+1] <= 2).all():
            charge_end_time = current_window['time'].iloc[-1]
            start_times2.append(charge_end_time + datetime.timedelta(minutes=buffer_minute)) 
            i = i + window_size
        else:
            i = i + 1       

    return start_times2, end_times2

def visualize_charging(df_battery, start_times, end_times, start_times2, end_times2): 
    """Visualize two methods of charging times."""
    x = df_battery['time']
    y1 = df_battery['state']
    y2 = df_battery['level']
    fig, ax = plt.subplots()
    ax.plot(x, y1, label='state')
    ax.plot(x, y2, label='level')
    ax.scatter(start_times, [100] * len(start_times), color='red', marker='o', s=5)
    ax.scatter(end_times, [35] * len(end_times), color='red', marker='o', s=5)
    ax.scatter(start_times2, [95] * len(start_times2), color='green', marker='o', s=5)
    ax.scatter(end_times2, [20] * len(end_times2), color='green', marker='o', s=5)
    
    plt.xlabel('X-axis', fontsize=6)  # Adjust 'fontsize' parameter for x-label size
    plt.show()
    
def main(): 
    """We calculated the charging time using two approaches."""
    start_time, end_time = read_times(time_file, patient_ID)
    # Approach 1: using Status     
    df_battery, start_times, end_times =  charging_timev1(battery_folder, start_time, end_time)
    
    # Approach 2: using level
    start_times2, end_times2 = charging_timev2(df_battery, start_time, end_time)

    visualize_charging(df_battery, start_times, end_times, start_times2, end_times2)
    
    # Check if each end time is later than start time
    for i in range(len(start_times2)):
        if end_times2[i] < start_times2[i]:
            print("*"*20)
            print("Battery charging time has problem", i, "*start*", start_times2[i], "*end*", end_times2[i])
            print("*"*20)
            
    # Adjust if the end time is earlier than the start time
    # start_times2.pop(1)
    # end_times2.insert(5, end_times[5])  # at index 5 (6th position)
    # start_times2.insert(2, pd.to_datetime('2023-06-12 22:06:54.372000'))
    # datetime.strptime('2023-07-02 18:49:28', "%Y-%m-%d %H:%M:%S") > datetime.strptime('2023-07-02 21:08:05', "%Y-%m-%d %H:%M:%S")
    # start_times = start_times.tolist()
    
    miss_battery = pd.DataFrame({'Start': start_times2, 'End': end_times2})
            
    miss_battery.to_csv(main_output, index=False)
    

if __name__ == "__main__":
    main()

