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
This script calculate questionnaire scores. 

Input:
- input_folder: Path where questionnaire files stores.
- baseline_file: File path which basic information of the participants stores.

Output:
- output_folder: Folder path which stores questionnaire scores.
- baseline_output: File path which stores age, gender of the participants.
- MEQ_score_file: File path which stores MEQ scores.
"""

import pandas as pd
from matplotlib import pyplot as plt
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from scipy.stats import shapiro, anderson, pearsonr, spearmanr
from utils.read_file import load_config
from utils.error_handling import folderErrorHandling


############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
input_root, qq_path, baseline_file = (
    config.get(key, "") for key in [
        "input_root", "raw_questionnaire_folder", "baseline_file" 
    ]
)
input_folder = os.path.join(input_root, qq_path)


# Output folder and files
output_root, output_path, baseline_output, MEQ_score_file = (
    config.get(key, "") for key in [
        "output_root", "Q_score_folder", "baseline_output", "MEQ_score_file"
    ]
)

output_folder = os.path.join(output_root, output_path)

folderErrorHandling(output_folder)

# Parameters
study_year = 2023
    
def MEQ_calculation(folder):
    """Calculate average MEQ score from 3 times of recording."""
    MEQs = []
    
    for idx in range(1,4):
        file_path = os.path.join(folder, 'Q'+ str(idx) +'.csv')
        df = pd.read_csv(os.path.join(file_path))
        
        points_columns = [f"Points.{i}" for i in range(1, 19)]  # Assuming columns are named "points1", "points2", ..., "points5"
        
        MEQ = df[points_columns]
        MEQ.insert(loc=0, column='Points.0', value=df['Points'])
        MEQ.insert(loc=0, column='SUM', value= MEQ.sum(axis=1))
        MEQ.insert(loc=0, column='ID', value= df['Participant ID'])
    
        MEQs.append(MEQ)
        
    sum_MEQ = pd.DataFrame({'ID': MEQs[0]['ID'],'MEQ1': MEQs[0]['SUM'], 'MEQ2': MEQs[1]['SUM'], 'MEQ3': MEQs[2]['SUM']})
    sum_MEQ['MEQ'] = sum_MEQ[['MEQ1', 'MEQ2', 'MEQ3']].mean(axis=1).round(2)
    
    return sum_MEQ


def baseline_summary(df_baseline):
    """Keep relevant information from the baseline questionnaire."""
    
    for col in df_baseline.columns:
        if "birth year" in col.lower():
            df_baseline[col] = pd.to_numeric(df_baseline[col], errors='coerce').astype('Int64')
            df_baseline['Age'] = 2023 - df_baseline[col]
            break  
        
    #df_baseline.iloc[:, 4] = df_baseline.iloc[:, 4].astype(int)
    #df_baseline['Age'] = study_year - df_baseline.iloc[:, 4] 
    df_baseline['BMI'] = df_baseline['Weight (kg)'] / ((df_baseline['Height (cm)']/100) ** 2)
    adapt_names = {'Participant ID': 'ID', 'Biological sex': 'Gender', 'What is your main racial group?': 'Racial', 'Which of the following best describes your main work status over the past 12 months?': 'Job', 'If you are employed (or a student), how much hours do you work (study) per week?': 'Working hour'}
    df_baseline = df_baseline.rename(columns=adapt_names)
    
    df_baseline.rename(columns=lambda x: "Coffee Binary" if x.startswith("Do you drink coffee, tea") else x, inplace=True)
    df_baseline.rename(columns=lambda x: "Coffee Amount" if x.startswith("How many of those do you consume on average on a daily basis? (Quantities are for approximate purposes only and do not have to be exact) Coffee") else x, inplace=True)
    df_baseline.rename(columns=lambda x: "Alcohol Amount" if x.startswith("During the past 7 days") else x, inplace=True)
    
    # Prefix to remove
    prefix = 'At least 1 row is required in this question type. = '
    # Remove prefix from all columns
    df_baseline = df_baseline.applymap(lambda x: x.replace(prefix, '') if isinstance(x, str) else x)
    
    counts = df_baseline['Coffee Binary'].value_counts()
    for value, count in counts.items():
        print(f"Value: {value}, Count: {count}")
    print("*"*10)

    baseline_long = df_baseline[['ID', 'Age', 'Gender', "Racial", "Job", "Working hour", "BMI", "Coffee Binary", "Coffee Amount", "Alcohol Amount"]]

    baseline_short = df_baseline[['ID', 'Age', 'Gender', "Racial", "Job", "Working hour", "BMI"]]
    return baseline_short, baseline_long


def main():
    df_baseline = pd.read_csv(os.path.join(input_folder, baseline_file))    
    
    baseline_short, baseline_long = baseline_summary(df_baseline)

    MEQ_score = MEQ_calculation(input_folder)
    
    baseline_short.to_csv(os.path.join(output_folder, baseline_output), index=False)
    MEQ_score.to_csv(os.path.join(output_folder, MEQ_score_file), index=False)
     
    print(MEQ_score['MEQ'].describe())

    
if __name__ == "__main__":
    main()
