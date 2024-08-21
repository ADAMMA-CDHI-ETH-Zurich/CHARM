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
This script calculate correlation between MEQ and CR metrics (including cosinor and non-parametric metrics) for all individuals. 

Input:
- input_q_folder: Folder path which stores questionnaire scores.
- MEQ_score_file: File path which stores MEQ scores.
- input_CR_folder: Path where CR metrics stores.
- CR_model_file: File path which stores cosinor metrics for all individuals.
- np_file: File path which stores non-parametric metrics for all individuals.

Output:
- output_folder: Folder path which stores CR related files.
- output_csv: File path which stores correlation between MEQ and CR metrics.
- fig_folder_path: Folder stores correlation between acrophase and MEQ.
"""


import pandas as pd
from matplotlib import pyplot as plt
import os
import sys
from scipy.stats import pearsonr
import seaborn as sns
import numpy as np
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.read_file import load_config
from utils.error_handling import folderErrorHandling


############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
root, input_q_path, MEQ_score_file, CR_path, CR_model_file, np_file = (
    config.get(key, "") for key in [
        "output_root", "Q_score_folder", "MEQ_score_file", "CR_folder", "CR_model_file", "CR_non_para_file"
    ]
)

input_q_folder = os.path.join(root, input_q_path)
input_CR_folder = os.path.join(root, CR_path)

# Output folder and files
output_csv, fig_folder_path = (
    config.get(key, "") for key in [
        "CR_MEQ_corr", "fig_folder_path"
    ]
)

output_folder = os.path.join(root, CR_path)

folderErrorHandling(output_folder)
fig_folder = os.path.join(output_folder, fig_folder_path)
folderErrorHandling(fig_folder)
fig_file_path = os.path.join(fig_folder, "Acrophase_MEQ.png")

# Strings
str_Acti, str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4 = (
    config.get(key, "") for key in [
        "str_Acti", "str_Watch",  "str_CBT", "str_SkinT", "str_HR", "str_HRV1", "str_HRV2", "str_HRV3", "str_HRV4"
    ]
)
sensors = [str_Acti, str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4]


def correlation_group(df, x, y, sensor, group, xlabel):
    """Plot correlation between acrophase of 3 groups (activity, temperature, heart) with MEQ scores."""
    plt.style.use('seaborn-v0_8-colorblind')
    sns.set(font_scale=0.65)  # Adjust the font scale 
    g = sns.lmplot(data=df, x=x, y=y, hue=sensor, col=group, scatter_kws={"s": 1}, line_kws={'alpha': 0.9, 'linewidth':1}, legend=False)
    g.set(xlabel=xlabel)
    g.fig.set_size_inches(7, 2.5)  # Adjust the width and height 

    group_names = df[group].unique()
    # For each Group create a seperate panel
    for i, ax in enumerate(g.axes.flat):
        ax.set_facecolor('none')  # Transparent background
        
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        
        ax.spines['bottom'].set_linewidth(0.3)
        ax.spines['left'].set_linewidth(0.3)
        
        # Obtain sensor names in the current group
        which_group = df[df[group] == group_names[i]]
        group_sensors = which_group[sensor].unique()
        
        # For each sensor in the current group, calculate correlation and plot 
        for sensor_idx, which_sensor in enumerate(group_sensors):
            #print(which_sensor)
            sensor_df = df[df[sensor] == which_sensor]
            r, pvalue = pearsonr(sensor_df[x], sensor_df[y])
            #print(r, pvalue)
            p_num = 3 if pvalue < 0.001 else (2 if pvalue < 0.01 else (1 if pvalue < 0.05 else 0))
            ax.collections[sensor_idx*2].set_label(f'{which_sensor} : {r:.4f}'+"*"*int(p_num))
        
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    g.savefig(os.path.join(fig_file_path), dpi=300)


def calculate_correlation(results, model, metrics, sensors, ref_name, meq_score, model_type):
    """Calculate correlation between CR metrics with MEQ scores."""
    for which_metrics in metrics:
        for which_sensor in sensors:
            print("="*30)
            print(which_sensor, which_metrics)
          
            metrics_chosen = model[model['test' if model_type =="CR" else 'Measurement'] == which_sensor]
            metrics_chosen = metrics_chosen.reset_index(drop=True)

            data = pd.DataFrame({'ID': meq_score['ID'], 
                                 f"{ref_name}": meq_score[ref_name], 
                                 f"{which_metrics}_{which_sensor}": metrics_chosen[which_metrics]})


            corr_coeff, p_val = pearsonr(data.iloc[:, 1], data.iloc[:, 2])

            column_names = ['CR Metrics', 'Sensor', 'Correlation coefficient', 'p-value (corr)']
            
            data = [which_metrics, which_sensor, corr_coeff, p_val]            
            df_result = pd.DataFrame(data, index=column_names).T

            results = pd.concat([results, df_result], ignore_index=True)
    
    return results


def main():
    ref_name = 'MEQ'
    meq_score = pd.read_csv(os.path.join(input_q_folder, MEQ_score_file))
    cos_model = pd.read_csv(os.path.join(input_CR_folder, CR_model_file))
    np_model = pd.read_csv(os.path.join(input_CR_folder, np_file))
    
    cos_metrics = ['amplitude', 'time', 'mesor'] # time is the 24-hour representation of the acrophase
    np_metrics = ['IS', 'IV', 'M10', 'L5', 'RA']
    
    results = pd.DataFrame()
    results = calculate_correlation(results, cos_model, cos_metrics, sensors, ref_name, meq_score, model_type="CR")
    results = calculate_correlation(results, np_model, np_metrics, sensors, ref_name, meq_score, model_type="NP")
    results.to_csv(os.path.join(output_folder, output_csv), index=False)
    
    ### Plot the correlation between MEQ and acrophase
    cosinor_metrics_plot = "time" # Acrophase representation in 24 hour
    sensor_type = "Sensor Data"
    cos_model.rename(columns={"test": sensor_type}, inplace=True)    
    df_merged = pd.merge(cos_model, meq_score)
    data_acro = df_merged[['ID', sensor_type, cosinor_metrics_plot, ref_name]]
    data_acro['Group'] = np.where(data_acro[sensor_type].isin([str_Acti, str_Watch]), 'Activity', np.where(data_acro[sensor_type].isin([str_CBT, str_SkinT]), 'Temperature', 'Heart'))
    
    correlation_group(df=data_acro, x=cosinor_metrics_plot, y=ref_name, sensor=sensor_type, group="Group", xlabel='Acrophase [h]')


if __name__ == "__main__":
    main()






