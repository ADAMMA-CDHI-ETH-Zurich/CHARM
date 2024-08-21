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
This script calculates non-parametric metrics (i.e., IS, IV, M10, L5, RA) from the sensor data.

Input:
- input_folder: Path where sensor data stores
- ActiAC_file: File where activity counts of Actigraph stores.
- WatchAC_file: File where activity counts of the smartwatch stores.
- HR_file: File where heart rate obtained from the smartwatch stores.
- HRV_file: File where heart rate variablity stores.
- CBT_file: File where core and skin temperature stores.

Output:
- output_folder: Folder path which stores calculated non-parametric metrics.
- output_csv: File path which stores cosnon-parametric metrics for all individuals.
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import importlib
# Load the module
Cosinor_metrics = importlib.import_module("01_Cosinor_metrics")
read_sensor_data = getattr(Cosinor_metrics, "read_sensor_data")
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
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
output_path, output_csv = (
    config.get(key, "") for key in [
        "CR_folder", "CR_non_para_file"
    ]
)

output_folder = os.path.join(root, output_path)


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


def non_parametrics(data, which_measure):
    """Calculate non-parametric metrics for the specific data from each individual."""
    data.set_index(str_time, inplace=True)

    d_fraction = data.groupby([data.index.hour, data.index.minute]).mean().var()

    d_daily = data.var()
    d_across = data.diff(1).pow(2).mean()

    IS = d_fraction / d_daily
    IV = d_across / d_daily
    
    hourly_activity = data.groupby(data.index.hour)[which_measure].mean()
    M10 = hourly_activity.nlargest(10).mean()
    L5 = hourly_activity.nsmallest(5).mean()
    RA = (M10 - L5) / (M10 + L5)
    
    return [which_measure, IS[0], IV[0], M10, L5, RA]


def scale_measure(df, scaler):
    """Scale the selected data."""
    df[df.columns[-1]] = scaler.fit_transform(df[[df.columns[-1]]])
    return df


def main():
    non_parametric_whole = pd.DataFrame()

    for patient_ID in patient_list:
        print("="*30)
        print(patient_ID)
        
        sensor_data = read_sensor_data(patient_ID)
        
        #sensor_data_agg = aggregate_df(sensor_data)
    
        scaler = MinMaxScaler()
    
        sensor_data_scaled = [scale_measure(df, scaler) for df in sensor_data]
    
        non_para_results = [non_parametrics(df, df.columns[-1]) for df in sensor_data_scaled]
        non_parametric_p = pd.DataFrame(non_para_results, columns=['Measurement', 'IS', 'IV', 'M10', 'L5', 'RA'])
        non_parametric_p.insert(loc=0, column='ID', value=patient_ID)
    
        non_parametric_whole = pd.concat([non_parametric_whole, non_parametric_p], ignore_index=True)
    
    non_parametric_whole.to_csv(os.path.join(output_folder, output_csv), index=False)
    
    
    
if __name__ == "__main__":
    main()