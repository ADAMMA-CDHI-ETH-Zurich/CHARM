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


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import sys
import csv
from utils.read_file import acc_csv
from utils.visualization import bland_altman_AC, regression_AC, scatter, stas_diff
from utils.error_handling import folderErrorHandling
from algorithms.counts_cal_ import calculate_AC, remove_both_non_wearing, remove_single_non_wearing

class ActivityCounts:
    """ActivityCounts is a class to calculate Activity counts and to compare."""
    def __init__(self, input_folder_p, patient_ID, times, time_folder_p=None, output_folder_p=None, fig_folder_path=None, stats_folder=None, watch_folder_path=None, 
                 acti_folder_path=None, str_Acti=None, str_Watch=None, str_time=None, counts_file_path=None, sleep_file_name=None,
                 charging_list=None, both_no_wear_list=None, single_no_wear_list=None):
        print("*"*20)
        print(patient_ID)
        print("*"*20, '\n')
        self.input_folder = input_folder_p
        self.time_folder = time_folder_p
        
        if watch_folder_path is not None:
            self.watch_input_folder = os.path.join(self.input_folder, watch_folder_path)
        
        if acti_folder_path is not None:
            self.acti_input_folder = os.path.join(self.input_folder, acti_folder_path)

        self.output_folder = output_folder_p
        self.stats_folder = stats_folder
        self.fig_folder_path = fig_folder_path

        self.start_times = pd.to_datetime(times['Start'])
        self.end_times = pd.to_datetime(times['End'])
        
        self.patient_ID = patient_ID
        self.str_Acti = str_Acti
        self.str_Watch = str_Watch
        self.str_time = str_time
        self.row_data = [patient_ID]  
        
        self.counts_file_path = counts_file_path
        self.sleep_file_name = sleep_file_name
        
        self.charging_list=charging_list
        self.both_no_wear_list= both_no_wear_list
        self.single_no_wear_list= single_no_wear_list

    def process(self):
        """Obtain Acitivity counts from Actigraph, caculate Activity counts from watch or read the counts if already exist, merge two Activity counts, remove invalid time and compare."""
        self.get_ActiAC()
        self.folder_check()
        self.caculate_WatchAC()
        #self.read_WatchAC()
        self.merge_ACs()
        self.clean_ACs()
        self.visulization()

    def folder_check(self):
        """Check if the folder path is valid."""
        if self.output_folder is not None:
            folderErrorHandling(self.output_folder)
            self.fig_folder = os.path.join(self.output_folder, self.fig_folder_path)
            folderErrorHandling(self.fig_folder)
        else:
            print("Output Folder is None. Please assign a valid path.")
        if self.stats_folder is not None:
            folderErrorHandling(self.stats_folder)
        else:
            print("Stats Folder is None. Please assign a valid path.")
        
    def get_ActiAC(self):  
        """Read Acitivity counts from Actigraph aggregated file."""
        csv_path = os.path.join(self.acti_input_folder, self.patient_ID + ".csv")
        self.ActiAC = acc_csv(csv_path, self.start_times.iloc[0], self.end_times.iloc[-1]) 
        
    def caculate_WatchAC(self):
        """Calculate Smartwatch activity counts for smartwatch."""
        self.WatchAC = calculate_AC(self.watch_input_folder, self.output_folder, self.start_times, self.end_times, self.counts_file_path)

    def read_WatchAC(self):
        """Read Smartwatch activity counts if already calculated."""
        self.WatchAC =pd.read_csv(os.path.join(self.output_folder, self.str_Watch+'.csv.gz'))
    
    def merge_ACs(self):
        """Merge Activity counts of Smartwatch and Actigraph."""
        self.ActiAC[self.str_time] = pd.to_datetime(self.ActiAC[self.str_time])
        self.WatchAC[self.str_time] = pd.to_datetime(self.WatchAC[self.str_time])
        self.data = pd.merge(self.WatchAC, self.ActiAC, on=self.str_time, how='inner', suffixes=(self.str_Watch, self.str_Acti)).reset_index(drop=True)
        self.data.rename(columns={'AC'+self.str_Watch: self.str_Watch, 'AC'+self.str_Acti: self.str_Acti}, inplace=True)
        self.data = self.data[[self.str_time, self.str_Acti, self.str_Watch]]
        self.data['diff'] = self.data[self.str_Watch] - self.data[self.str_Acti]
        self.data['average'] = (self.data[self.str_Watch] + self.data[self.str_Acti]) / 2
                
    def clean_ACs(self):
        """Remove charging time and non-wearing time."""
        pct_charging = round((len(self.ActiAC) - len(self.data)) / len(self.ActiAC) * 100, 2)
        print("Removing charging time (time no file from the watch)", pct_charging, "%\n")
        
        sleep_file = os.path.join(self.time_folder, self.sleep_file_name)
        
        if self.time_folder is not None:
            pct_both_no_wear, rm_data = remove_both_non_wearing(self.data, sleep_file, self.str_Watch, self.str_Acti)
            print("Removing time when both devices were not wearing", pct_both_no_wear, "%\n")
       
            pct_single_no_wear, final_data = remove_single_non_wearing(rm_data)
            print("Removing time when single device was not wearing", pct_single_no_wear, "%\n")
    
            self.final_data = final_data.copy()
            self.final_data['diff'] = self.final_data[self.str_Watch] - self.final_data[self.str_Acti]
            self.final_data['average'] = (self.final_data[self.str_Watch] + self.final_data[self.str_Acti]) / 2
            
            final_data.to_csv(os.path.join(self.output_folder, 'ACs.csv.gz'), index=False)
            
            self.charging_list.append(pct_charging)
            self.both_no_wear_list.append(pct_both_no_wear)
            self.single_no_wear_list.append(pct_single_no_wear)
                
        else:
            print("Time Folder is None. Please assign a valid path.")

    def visulization(self):
        """Visulization and compare ActiAC and WatchAC."""
        mae, rmse, t_stat, p_t = stas_diff(self.final_data[self.str_Watch], self.final_data[self.str_Acti])
        with plt.style.context('seaborn'):
            mean_diff, loa_lower, loa_upper = bland_altman_AC(self.final_data, plot_save=True, fig_folder=self.fig_folder)
            corr_coeff, p_corr = regression_AC(self.final_data[self.str_Watch], self.final_data[self.str_Acti], plot_save=True, fig_folder=self.fig_folder)
            scatter(self.final_data[self.str_time], self.final_data[self.str_Watch], self.final_data[self.str_Acti], self.str_Watch, self.str_Acti, plot_save=True, fig_folder=self.fig_folder)
        self.row_data.extend([mae, rmse, mean_diff] + [[loa_lower, loa_upper]] + [t_stat, p_t, corr_coeff, p_corr])



