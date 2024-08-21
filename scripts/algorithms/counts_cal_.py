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
from agcounts.extract import get_counts
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from utils.read_file import read_watch_acc_folder, sleep_csv, check_both_zero


def get_counts_csv(
    raw,
    freq: int,
    sampling_freq: str,
    epoch: int,
    fast: bool = True,
    verbose: bool = False,
    time_column: str = None,
):
    """Adapt from "get_counts_csv" in the package agcounts. It calculates activity counts based on 3-axis acceleration data."""
    raw = raw.dropna()
    # Find the missing time and fill 0 for the missing time
    rounded_start_time = raw[time_column].min().floor('T')
    rounded_end_time = raw[time_column].max().ceil('T')
    
    idx = pd.date_range(start=rounded_start_time, end=rounded_end_time, freq=sampling_freq, inclusive='left')
    idx_s = idx.strftime('%Y-%m-%d %H:%M:%S.%f')
    idx_s = pd.to_datetime(idx_s)
    missing_time = set(idx_s) - set(raw[time_column])
    
    s = pd.DataFrame({time_column: pd.to_datetime(list(missing_time))})
    
    s = s.assign(X=0.0, Y=0.0, Z=0.0)
    df = pd.concat([raw,s], axis=0).sort_values(by=[time_column]).reset_index(drop=True)
    
    if time_column is not None:
        ts = df[time_column]
        ts = pd.to_datetime(ts)
        time_freq = str(epoch) + "S"
        ts = ts.dt.floor(time_freq)
        ts = ts.unique()
        ts = pd.DataFrame(ts, columns=[time_column])
    
    df = df[["X", "Y", "Z"]]

    if verbose:
        print("Converting to array", flush=True)
    df = np.array(df)
    if verbose:
        print("Getting Counts", flush=True)
    counts = get_counts(df, freq=freq, epoch=epoch, fast=fast, verbose=verbose)
    del df
    counts = pd.DataFrame(counts, columns=["Axis1", "Axis2", "Axis3"])
    counts["AC"] = (counts["Axis1"] ** 2 + counts["Axis2"] ** 2 + counts["Axis3"] ** 2) ** 0.5
    ts = ts[0 : counts.shape[0]]
    if time_column is not None:
        counts = pd.concat([ts, counts], axis=1)
    return counts, s


def calculate_AC(samsung_folder, output_folder_p, start_times, end_times, counts_file_path):
    """Calculate AC for each wearing duration and merge results from all durations into one."""
    counts_watch = pd.DataFrame()
    df_samsung_whole = pd.DataFrame()
    
    for i, el in tqdm(enumerate(start_times.to_numpy())):
        print("Timeslot", str(i+1), " in ", len(start_times))
        start_time = start_times[i] 
        end_time = end_times[i]
    
        try:
            df_samsung = read_watch_acc_folder(samsung_folder, start_time, end_time)
        except Exception as e:
            print("An exception occurred in complete_samsung:", e)
            continue
        
        if len(df_samsung) != 0: 
            df_samsung_whole = pd.concat([df_samsung_whole, df_samsung])
            
            counts_sub, miss = get_counts_csv(
                raw=df_samsung, freq=50, sampling_freq="20ms", epoch=60, verbose=False, time_column="time"
            )
            
            counts_watch = pd.concat([counts_watch, counts_sub])
    
    counts_watch.to_csv(os.path.join(output_folder_p, counts_file_path), index=False)
    return counts_watch
    

def remove_both_non_wearing(data, sleep_file, str_Watch, str_Acti):
    """Remove windows when both devices activities are zero for 15 minutes outside sleep."""
    rm_data = pd.DataFrame(columns=data.columns)
    nonwear_start = []
    
    # Read each pair of sleep start and end time
    sleep_start, sleep_end = sleep_csv(sleep_file)
    
    for window_start in pd.date_range(start=data['time'].min(), end=data['time'].max() - pd.Timedelta(minutes=15), freq='15T'):
        insleep = False
        window_end = window_start + pd.Timedelta(minutes=15)
        window = data[(data['time'] >= window_start) & (data['time'] < window_end)]
        
        for count, _ in enumerate(sleep_start):
            t0 = sleep_start[count]
            t1 = sleep_end[count]
            if (window_start > t0) & (window_start < t1):
                insleep = True
                break
            
        if insleep == False and len(window) >= 2:
            if check_both_zero(window.iloc[:-1], str_Watch, str_Acti):
                nonwear_start.append(window.iloc[0]['time'])
            else:
                rm_data = pd.concat([rm_data, window])
        else:
            rm_data = pd.concat([rm_data, window])
    
    
    pct_both_non_wear = round((len(data) - len(rm_data)) / len(data) * 100, 2)

    return pct_both_non_wear, rm_data


def remove_single_non_wearing(data):
    """Remove when difference of two ACs is larger than twice of the average ACs or when difference of ACs less than 500."""
    condition_1 = (data['diff'] < 2 * data['average']) & (data['diff'] > -2 * data['average'])
    condition_2 = (abs(data['diff']) < 500)
    rm_data = data[condition_1 | condition_2]
    
    pct_single_non_wear = round((len(data) - len(rm_data)) / len(data) * 100, 2)
    
    return pct_single_non_wear, rm_data
