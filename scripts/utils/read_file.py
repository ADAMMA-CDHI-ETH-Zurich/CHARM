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
This script include several functions to read different files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import csv
from datetime import timedelta
import os
import numpy as np
import json

def check_both_zero(df_subset, str1, str2):
    """Return true if both are non zero."""
    return df_subset[str1].eq(0).all() and df_subset[str2].eq(0).all()


def parse_datetime(datetime_str):
    """Check if the input string contains minutes and seconds, if no, append "00:00:00" and parse datetime string."""
    try:
        datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        datetime_str = datetime_str + ' 00:00:00'

    dttime = pd.to_datetime(datetime_str)
    return dttime 


def sleep_csv(file):
    """Read sleep time."""
    sleep_times= pd.read_csv(file)  
    sleep_start = sleep_times['in_bed_time'].apply(parse_datetime)
    sleep_end = sleep_times['out_bed_time'].apply(parse_datetime)
    
    return sleep_start, sleep_end



def acc_csv(file_path, start_time, end_time):
    """Read Activity counts from Actigraph file within defined timeframe."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.replace(' ', '')
    
    for index, row in df.iterrows():
        try:
            parsed_date = datetime.datetime.strptime(row['Date'], "%m.%d.%y")
            converted_date = parsed_date.strftime("%m/%d/%Y")
            df.loc[index, 'Date'] = converted_date
        except ValueError:
            continue
    
    df['time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # exchange values in Axis 1 and Axis 2
    df['tmp'] = df['Axis1']
    df['Axis1'] = df['Axis2']
    df['Axis2'] = df['tmp']
    df['AC'] = df['VectorMagnitude']

    df = df[['time', 'Axis1', 'Axis2', 'Axis3', 'AC']]
    
    df = df[(df["time"] >= start_time) & (df["time"] < end_time)]
    df = df.reset_index(drop=True)
    return df



def acc_watch_csv(file_path):  
    """Read each raw acceleration file from smartwatch file."""
    df = pd.read_csv(file_path, on_bad_lines='skip')
    
    # delete bad lines
    df = df[df['UnixTimestamp'].notna()]
    df = df.apply(pd.to_numeric, errors='coerce')
    df['time'] = df['UnixTimestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000) if pd.notnull(x) else x)
    df.dropna(subset=['time'], inplace=True)
    
    # If DataFrame has less than 50 rows, skip
    if len(df) < 50:
        return
    
    # delete time outside of this hour
    start_time = datetime.datetime.strptime(file_path.rsplit(".", 1)[0].rsplit('/', 1)[1], "%d.%m.%y_%H")    
    end_time = start_time + timedelta(hours=1)    
    df = df.drop(df[~df['time'].between(start_time, end_time)].index)
    
    df = df.reset_index(drop=True)

    df['time'] = pd.to_datetime(df['time'])
    
    # Adapt the acceleration unit to mG
    df['X'], df['Y'], df['Z'] = df['x']/4096, df['y']/4096, df['z']/4096
    df = df[['time', 'X', 'Y', 'Z']]
    
    # Upsample data from 25Hz to 50Hz 
    rounded_start_time = df['time'].min().floor('T')
    rounded_end_time = df['time'].max().ceil('T')
    idx = pd.date_range(start=rounded_start_time, end=rounded_end_time, freq="20ms", inclusive='left')
    
    df = df.set_index('time')
    df = df.loc[~df.index.duplicated(), :]
    df.index = df.index.sort_values()
    
    mapped_df = df.reindex(idx, method='nearest')
    mapped_df = mapped_df.reset_index()
    mapped_df = mapped_df.rename(columns={"index": "time"})

    return mapped_df
    

def read_watch_acc_folder(folder_path, start_time, end_time):
    """List and read all related acceleration files and combine as one dataframe."""
    file_names = os.listdir(folder_path)
    csv_names = [file_name for file_name in file_names if file_name.endswith('.csv')]
    sorted_csv_names = sorted(csv_names, key=lambda x: datetime.datetime.strptime(x.rsplit('.', 1)[0], "%d.%m.%y_%H"), reverse=False)
    
    df_samsung = pd.DataFrame()
    for file in sorted_csv_names:
        timestamp = datetime.datetime.strptime(file, "%d.%m.%y_%H.csv")
        if start_time - timedelta(hours=1) < timestamp < end_time:
            file_path = os.path.join(folder_path, file)
            try:
                df = acc_watch_csv(file_path)
                df_samsung = pd.concat([df_samsung, df])
            except Exception as e:
                print(file_path)
                print("An exception occurred in read_samsung_csv:", e)
                continue
        elif timestamp > end_time:
            break
    filtered_df = df_samsung[(df_samsung['time'] >= start_time) & (df_samsung['time'] < end_time)]
    return filtered_df


def hr_csv(file_path):
    """Read each valid heart rate data and RR interval data."""
    try:
        df = pd.read_csv(file_path)
        df = df[df['UnixTimestamp'].notna()]
        df = df.apply(pd.to_numeric, errors='coerce')
        df['time'] = df['UnixTimestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000) if pd.notnull(x) else x)
        df['time'] = pd.to_datetime(df['time'])
        
        if len(df) < 50:
            return
        
        start_time = datetime.datetime.strptime(file_path.rsplit(".", 1)[0].rsplit('/', 1)[1], "%d.%m.%y_%H")    
        end_time = start_time + timedelta(hours=1)    
        df = df.drop(df[~df['time'].between(start_time, end_time)].index)
        
        df.loc[df['status'] != 1, 'hr'] = np.nan
        df.dropna(subset=['time'], inplace=True)    
        df = df.reset_index(drop=True)

        df = df[['time', 'hr', 'hrIbi']]
        df = df.dropna(subset=['hr'])

        df['time'] = df['time'].apply(lambda dt: dt.replace(microsecond=0))
        return df
    except Exception as e:
        print(e)
        print(file_path)


def read_hr_folder(folder_path):
    """List and read all related heart rate files and combine as one dataframe."""
    csv_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.csv')])
    df_hr = pd.DataFrame()
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = hr_csv(file_path)            
            df_hr = pd.concat([df_hr, df], ignore_index=True)

        except pd.errors.ParserError:
            print(f"Error parsing file: {file_path}. Skipping the file.")
    df_hr = df_hr.rename(columns={'hr': 'HR'})
    return df_hr


def battery_csv(file_path):
    """Read each battery file."""
    try:
        df = pd.read_csv(file_path)
        df = df[df['UnixTimestamp'].notna()]
        df = df.apply(pd.to_numeric, errors='coerce')
        df['time'] = df['UnixTimestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000) if pd.notnull(x) else x)
        df.dropna(subset=['time'], inplace=True)
        df = df.reset_index(drop=True)
        df['time'] = pd.to_datetime(df['time'])
        return df
    except pd.errors.EmptyDataError:
       print (file_path, " is empty")
       return None
   

def core_csv(path, times, patient_ID):
    """Read the CSV file, and keep CBT and SkinT when the recorded data quality is over 2."""
    start_str = times.loc[times['ID'] == int(patient_ID), 'Start'].iloc[0]
    end_str = times.loc[times['ID'] == int(patient_ID), 'End'].iloc[0]
    
    try:
        start_time = datetime.datetime.strptime(start_str, "%d.%m.%y %H:%M")
        end_time = datetime.datetime.strptime(end_str, "%d.%m.%y %H:%M")
    except ValueError:
        start_time = datetime.datetime.strptime(start_str, "%d.%m.%Y %H:%M")
        end_time = datetime.datetime.strptime(end_str, "%d.%m.%Y %H:%M")

    deli = times.loc[times['ID'] == int(patient_ID), 'Delimiter'].iloc[0]
    time_format = times.loc[times['ID'] == int(patient_ID), 'Timeformat'].iloc[0]    

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=deli)
        first_row = next(reader)
        if any("SEP" in s for s in first_row):
            df = pd.read_csv(path, sep=deli, skiprows=1)
        else:
            df = pd.read_csv(path, sep=deli)

    if time_format == 1:
        df['DateTime'] = df['DateTime'].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M'))
        df['DateTime'] = pd.to_datetime(df['DateTime'])

    if time_format == 2:
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d.%m.%y %H:%M')
        df['DateTime'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M')


    df['CBT'] = df['CoreBodyTemp [C]']
    df['SkinT'] = df['SkinTemp [C]']
    df['qualityT'] = df['TempQuality [1(poor) to 4(excellent)]']
    df = df[(df["qualityT"] >= 3)]
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df[(df["DateTime"] >= start_time) & (df["DateTime"] < end_time)]
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(keep='last')

    df['time'] = df['DateTime'].dt.round('min')

    df = df[(df["CBT"] >= 0)]
    df = df[['time', 'CBT', 'qualityT', 'SkinT']]
    return df, start_time, end_time



def load_config(current_dir):
    """Load parameters defined in the config file."""
    config_path = os.path.join(current_dir, '..', 'config.json')
    with open(config_path) as f:
        return json.load(f)










