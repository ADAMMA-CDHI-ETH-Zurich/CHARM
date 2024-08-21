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
This script compares CR metrics (including cosinor and non-parametric metrics) for all individuals. 

Input:
- input_folder: Path where CR metrics stores.
- cosinor_file: File path which stores cosinor metrics for all individuals.
- np_file: File path which stores non-parametric metrics for all individuals.

Output:
- output_folder: Folder path which stores CR related files.
- output_csv: File path which stores comparison of CR metrics for all individuals.
"""

import os
import pandas as pd
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from scipy.stats import shapiro, anderson, pearsonr, spearmanr, wilcoxon
import numpy as np
import scipy.stats as stats
from utils.read_file import load_config


############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
root, input_path, cosinor_file, np_file = (
    config.get(key, "") for key in [
        "output_root","CR_folder", "CR_model_file", "CR_non_para_file"
    ]
)

data_folder = os.path.join(root, input_path)

# Output folder and files
output_csv, = (
    config.get(key, "") for key in [
        "CR_comparsion_file", 
    ]
)

#output_csv = os.path.join(data_folder, output_csv)


# Strings
str_time, str_Acti, str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4 = (
    config.get(key, "") for key in [
        "str_time", "str_Acti", "str_Watch",  "str_CBT", "str_SkinT", "str_HR", "str_HRV1", "str_HRV2", "str_HRV3", "str_HRV4"
    ]
)

# Use the metrics calculated from Actigraph as the reference, and the metrics calculated from other sensor data as the test data.
sensor_ref = str_Acti
sensor_to_test = [str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4]



def compare_metrics(data, which_sensor, which_metrics):
    """Calculate comparison metrics (e.g., MAE, RMSE, correlation) for the selected data."""
    y1 = data.iloc[:, 1]
    y2 = data.iloc[:, 2]
    
    mean_y1 = y1.mean()
    std_y1 = y1.std()
    
    mean_y2 = y2.mean()
    std_y2 = y2.std()
    
    quartiles_y1 = y1.quantile([0.25, 0.75])
    iqr_y1 = quartiles_y1[0.75] - quartiles_y1[0.25]
    
    quartiles_y2 = y2.quantile([0.25, 0.75])
    iqr_y2 = quartiles_y2[0.75] - quartiles_y2[0.25]

    y1_stat = str(round(mean_y1, 2)) + '(' + str(round(std_y1,2)) + ')' 
    y2_stat = str(round(mean_y2, 2)) + '(' + str(round(std_y2,2)) + ')' 

    y1_stat2 = str(round(y1.median(), 2)) + '(' + str(round(iqr_y1,2)) + ')' 
    y2_stat2 = str(round(y2.median(), 2)) + '(' + str(round(iqr_y2,2)) + ')'
    
    mae = np.mean(np.abs(y1 - y2))
    rmse = np.sqrt(np.mean((y1 - y2) ** 2))
    
    # Test if the data is normal distribution
    statistic, p_value = stats.shapiro(data)
    if p_value < 0.05:
        #print("Data is not normally distributed.")
        pass
    else:
        print("Data appears to be normally distributed.")
    
    # Wilcoxon's signed-rank test statistic
    t_stat, p_t = wilcoxon(y1, y2)

    #print(f"Wilcoxon's signed-rank test statistic: {statistic}")
    #print(f"P-value: {p_value}")
    
    #t_stat, p_t = stats.ttest_ind(y1, y2)
    #print("t-statistic:", t_stat)
    #print("p-value:", p_t)
        
    corr_coeff, p_val = pearsonr(y1, y2)
    #print("Correlation coefficient:", corr_coeff, p_val)
    
    result = [which_sensor, which_metrics, y1_stat, y2_stat, y1_stat2, y2_stat2, mae, rmse, round(t_stat, 4), round(p_t, 4), round(corr_coeff, 4), round(p_val, 4)]

    return result


def calculate_metrics(compare_results, model, metrics, sensors, sensor_ref, model_type):
    """Calculate comparison metrics for all metrics from all sensor data."""
    for which_metrics in metrics:
        for which_sensor in sensors:
            print("="*30)
            print(which_sensor, which_metrics)

            metrics_reference = model[model['test' if model_type =="CR" else 'Measurement'] == sensor_ref]
            metrics_chosen = model[model['test' if model_type =="CR" else 'Measurement'] == which_sensor]
            metrics_reference = metrics_reference.reset_index(drop=True)
            metrics_chosen = metrics_chosen.reset_index(drop=True)

            data = pd.DataFrame({'ID': metrics_reference['ID'], 
                                 f"{which_metrics}_ref": metrics_reference[which_metrics], 
                                 f"{which_metrics}_{which_sensor}": metrics_chosen[which_metrics]})

            result = compare_metrics(data, which_sensor, which_metrics)        

            column_names = ['CR Metrics', 'Sensor', 'Mean (SD) (ref)', 'Mean (SD) (test)', 'Median (IQR) (ref)', 
                            'Median (IQR) (test)', 'MAE', 'RMSE', 't-statistic', 'p-value [t]', 
                            'Correlation coefficient', 'p-value (corr)']
            df_result = pd.DataFrame([result], columns=column_names)

            compare_results = pd.concat([compare_results, df_result], ignore_index=True)
    
    return compare_results





def main():
    cos_model = pd.read_csv(os.path.join(data_folder, cosinor_file))
    np_model = pd.read_csv(os.path.join(data_folder, np_file))
    
    cos_metrics = ['amplitude', 'time', 'mesor'] # time is the 24-hour representation of the acrophase
    np_metrics = ['IS', 'IV', 'M10', 'L5', 'RA']
    
    
    compare_results = pd.DataFrame()
    
    compare_results = calculate_metrics(compare_results, cos_model, cos_metrics, sensor_to_test, sensor_ref, model_type="CR")
    compare_results = calculate_metrics(compare_results, np_model, np_metrics, sensor_to_test, sensor_ref, model_type="NP")
    
    compare_results.to_csv(os.path.join(data_folder, output_csv), index=False)
    
        
if __name__ == "__main__":
    main()