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
This script include several functions to  handle errors.
"""


import os
  
def inputErrorHandling(indicator, condition='empty'):
    """Check if input is correct"""
    try:
        while True:
            value = input(indicator)
            if condition == 'empty':
                if value.strip():  # Check if input is not empty
                    break
                else:
                    print("Input cannot be empty. Please try again.")
        return value
    except ValueError as e:
        print("Invalid input:", e)


        
def folderErrorHandling(folder_path):
    """Check if the folder exists, if not, create it"""
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            #print("Folder exists!")
            pass
        else:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
    except Exception as e:
        print("An error occurred:", e)


def fileErrorHandling(file_path):
    """Check if the file can be found"""
    try:
        with open(file_path, "r") as file:
            content = file.read()
    except FileNotFoundError:
        print("File not found.")
    except IOError as e:
        print("Error reading file:", e)
        