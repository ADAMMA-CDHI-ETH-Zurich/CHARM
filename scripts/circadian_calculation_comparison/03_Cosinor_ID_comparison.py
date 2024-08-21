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
This script compare cosinor metrics (i.e., amplitude, mesor, acrophase) from one sensor data between two individuals.

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
- output_csv: File path which stores comparison of cosinor metrics for two individuals.
"""


from CosinorPy import cosinor, cosinor1, cosinor_nonlin
import importlib
importlib.reload(cosinor)
import pandas as pd
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

Cosinor_metrics = importlib.import_module("01_Cosinor_metrics")
read_sensor_data = getattr(Cosinor_metrics, "read_sensor_data")
to_cosinor_format = getattr(Cosinor_metrics, "to_cosinor_format")
cosinor_metrics = getattr(Cosinor_metrics, "cosinor_metrics")
aggregate_df = getattr(Cosinor_metrics, "aggregate_df")

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
        "fig_folder_path", "CR_folder", "CR_model_folder", "CR_ID_comparsion_file"
    ]
)

output_folder = os.path.join(root, output_path)

fig_folder = os.path.join(output_folder, fig_folder_path)
folderErrorHandling(fig_folder)

# Parameters
str_time, str_Acti, str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4 = (
    config.get(key, "") for key in [
        "str_time", "str_Acti", "str_Watch",  "str_CBT", "str_SkinT", "str_HR", "str_HRV1", "str_HRV2", "str_HRV3", "str_HRV4"
    ]
)

str_to_test = [str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4]

# Take the participant need to be compared
patient_list = ['01', '02']

time_interval = 10 # Aggerate to 10T

cr_data_whole = [pd.DataFrame() for _ in range(9)]

which_cr_metric = 4 # Take CR metric HR

def main():
    
    for patient_ID in patient_list:
        print("="*30)
        print(patient_ID)
        
        sensor_data = read_sensor_data(patient_ID)
        
        sensor_data_agg = aggregate_df(sensor_data)

        cr_data = [to_cosinor_format(df, patient_ID) for df in sensor_data_agg]
    
        # Concatenate the circadian data of all participants together
        for i, df in enumerate(cr_data_whole):
            cr_data_whole[i] = pd.concat([df, cr_data[i]], ignore_index=True)
    
        
    for df in cr_data_whole:
        df.dropna(subset=['y'], inplace=True)
    
    
    pairs = (patient_list, )
    
    
    '''
    pairs = ()
    
    for i in range(len(patient_list)):
        for j in range(i + 1, len(patient_list)):
            pairs += ([patient_list[i], patient_list[j]],)
    '''
    
    cr_data_to_compare = cr_data_whole[which_cr_metric]
    
    cosinor_metrics(cr_data_to_compare, int(24*60/time_interval), pairs, plot_figure=fig_folder, compare_csv=os.path.join(output_folder, output_csv))
        


if __name__ == "__main__":
    main()
