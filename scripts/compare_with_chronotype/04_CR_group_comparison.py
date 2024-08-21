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
This script compare the circadian rhythm metrics for three chronotype. 

Input:
- input_q_folder: Folder path which stores questionnaire scores.
- MEQ_score_file: File path which stores MEQ scores.
- input_CR_folder: Path where CR metrics stores.
- CR_model_file: File path which stores cosinor metrics for all individuals.
- age_gender_file: File path which stores age and gender.

Output:
- output_folder: Folder path which stores CR related files.
- output_coe_csv: File path which stores correlation between predicted MEQ and acrophase.
- output_res_csv: File path which stores prediction model metrics between predicted MEQ and acrophase.
- output_coe_ag_csv: File path which stores correlation between predicted MEQ and acrophase after adding age and gender as control variables.
- output_res_ag_csv: File path which stores prediction model metrics between predicted MEQ and acrophase after adding age and gender as control variables.
- fig_folder_path: Folder stores correlation plot between predicted MEQ and acrophase.
"""


import pandas as pd
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import scipy.stats as stats
from utils.read_file import load_config

############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
root, input_q_path, MEQ_score_file, CR_path, CR_model_file, age_gender_file = (
    config.get(key, "") for key in [
        "output_root", "Q_score_folder", "MEQ_score_file", "CR_folder", "CR_model_file", "baseline_output"
    ]
)

input_q_folder = os.path.join(root, input_q_path)
input_CR_folder = os.path.join(root, CR_path)

# Output folder and files
output_csv, = (
    config.get(key, "") for key in [
        "Group_comparison_file"
    ]
)

output_folder = os.path.join(root, CR_path)


# Strings
str_ID, str_Acti, str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4 = (
    config.get(key, "") for key in [
        "str_ID", "str_Acti", "str_Watch",  "str_CBT", "str_SkinT", "str_HR", "str_HRV1", "str_HRV2", "str_HRV3", "str_HRV4"
    ]
)

ref_name = 'MEQ'
metrics_focus = 'time' # Acrophase representation in 24 h
sensors_focus = [str_Acti, str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV3]

# MEQ score to distinguish between Evening and Intermediate, and between Intermediate and Morning
EI_border = 42
IM_border = 58    



def med_iqr(df):
    """Calculate Median, IQR."""
    quartiles = df.quantile([0.25, 0.75])
    iqr = quartiles[0.75] - quartiles[0.25]
    return round(df.median(),2), round(iqr, 2)

def test_normality(df):
    """Test for normality (Shapiro-Wilk test), print if data is normal distribution."""
    statistic, p_value = stats.shapiro(df)
    if p_value < 0.05:
        pass
        #print("Data is not normally distributed.")
    else:
        pass
        #print("Data appears to be normally distributed.")
    
    
def test_normality_homogeneity(df1, df2, df3):
    """Test for normality and homogeneity ."""
    #Test for normality (Shapiro-Wilk test)
    [ test_normality(df) for df in [df1, df2, df3] ]

    # Test for homogeneity of variance (Levene's test), print if variance are not equal between groups
    statistic, p_value = stats.levene(df1, df2, df3)
    if p_value < 0.05:
        pass
        #print("Variances are not equal between groups.")
    else:
        pass
        #print("Variances appear to be equal between groups.")


def KWtest(df1, df2, df3):
    """Perform Kruskal–Wallis test for for group comparison."""
    f_statistic, p_value = stats.kruskal(df1, df2, df3)    
    return round(f_statistic, 2), round(p_value, 4)


def Wtest(df1, df2, df3):
    """Perform the Wilcoxon rank-sum test for pairwise group comparison."""
    stat1, p_value1 = stats.ranksums(df1, df2)
    stat2, p_value2 = stats.ranksums(df2, df3)
    stat3, p_value3 = stats.ranksums(df1, df3)
    return round(stat1, 2), round(p_value1, 4), round(stat2, 2), round(p_value2, 4), round(stat3, 2), round(p_value3, 4), 


def custom_describe(df):
    """Summary statistics for numerical and categorical columns"""
    
    # Column time: Median and IQR
    median_col1 = df['time'].median()
    iqr_col1 = df['time'].quantile(0.75) - df['time'].quantile(0.25)
    
    # Columns age, MEQ: Average and Standard Deviation
    avg_col2 = df['Age'].mean()
    std_col2 = df['Age'].std()
    
    avg_col3 = df['MEQ'].mean()
    std_col3 = df['MEQ'].std()
    
    # Gender: Percentage
    female_percentage = (df['Gender'].value_counts(normalize=True) * 100)['Female']
    
    num_rows = len(df)

    print(f"Acrophase: Median = {median_col1:.2f}, IQR = {iqr_col1:.2f}")
    print(f"Age: Average = {avg_col2:.2f}, Std = {std_col2:.2f}")
    print(f"MEQ: Average = {avg_col3:.2f}, Std = {std_col3:.2f}")
    print(f"Percentage of Females: {female_percentage:.2f}%")
    print(f"Number of samples: {num_rows:.2f}\n")

def main():
    meq_score = pd.read_csv(os.path.join(input_q_folder, MEQ_score_file))
    cos_model = pd.read_csv(os.path.join(input_CR_folder, CR_model_file))
    age_gender = pd.read_csv(os.path.join(input_q_folder, age_gender_file))

    output_col = ['Sensor', 'Kruskal–Wallis test', 'p-value (f)', 'Wilcoxon rank-sum test  (EI)', 'p-value (t)', 'Wilcoxon rank-sum test  (IM)', 'p-value (t)', 'Wilcoxon rank-sum test  (EM)', 'p-value (t)']
    print(meq_score[ref_name].describe())
    
    compare_results = pd.DataFrame()

    for which_sensor in sensors_focus:
        print(which_sensor, "*"*30)
        
        metrics_chosen = cos_model[cos_model['test'] == which_sensor]
        metrics_chosen = metrics_chosen.reset_index(drop=True)
        
        # Merge MEQ scores, acrophase and demographics (age, gender)
        data = pd.DataFrame({str_ID: meq_score[str_ID], ref_name: meq_score[ref_name], metrics_focus : metrics_chosen[metrics_focus]  })
        data_ag = pd.merge(data, age_gender, on=str_ID)
    
        # group1 - Evening, group 2 - Intermediate , group3 - Morning
        group1 = data_ag[data_ag[ref_name] < EI_border]
        group2 = data_ag[(data_ag[ref_name] >= EI_border) & (data_ag[ref_name] <= IM_border)]
        group3 = data_ag[data_ag[ref_name] > IM_border]
        
        # Acrophase for each group
        group1_acro, group2_acro, group3_acro = (group[metrics_focus] for group in [group1, group2, group3])
    
        [custom_describe(group) for group in [group1, group2, group3]]
        
        med1, iqr1 = med_iqr(group1_acro)
        
        # Test for normality and homogeneity 
        test_normality_homogeneity(group1_acro, group2_acro, group3_acro)
    
        # Comparison test
        f_KW, p_KW = KWtest(group1_acro, group2_acro, group3_acro)
        t_W1, p_W1, t_W2, p_W2, t_W3, p_W3 = Wtest(group1_acro, group2_acro, group3_acro)
        
        df_result = pd.DataFrame([[which_sensor, f_KW, p_KW, t_W1, p_W1, t_W2, p_W2, t_W3, p_W3]], columns=output_col)
        compare_results = pd.concat([compare_results, df_result], ignore_index=True)
        compare_results.to_csv(os.path.join(output_folder, output_csv), index=False)
    


if __name__ == "__main__":
    main()


