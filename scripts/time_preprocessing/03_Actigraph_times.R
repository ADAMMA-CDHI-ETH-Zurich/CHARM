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


##################################################################################
# Description:
# This script obtains Actigraph non-wearing and sleep time.

# Input:
# - root_folder: Path where study data stores
# - input_file: Path where Actigraph .agd file saves

# Output:
# - sleep_times: File path which stores sleeping time of Actigraph.
# - actigraph_non_wear_times: File stores non wearing time of Actigraph.
#################################################################################


remotes::install_github("dipetkov/actigraph.sleepr")
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library("actigraph.sleepr")
library(tibble)
library(dplyr)
library(lubridate)  # Load the lubridate package for working with date-time data

########### Create Sleep Time from Actigraph counts data

root_folder <- "/Users/fanwufan/Documents/PhD/Research/CROCO/StudyData/raw_data"
output_folder <- "/Users/fanwufan/Documents/PhD/Research/CROCO/Results/Time"

patient_list <- sprintf("%02d", seq(1, 38)[!(seq(1, 38) %in% c(30, 36))])
print(patient_list)

for (patient_ID in patient_list) {
  input_file <- file.path(root_folder, patient_ID, 'Actigraph', paste0(patient_ID, '.agd'))
  output_folder_p <- file.path(output_folder, patient_ID)
  
  agdb_10s <- read_agd(input_file)

  agdb_10s <- agdb_10s %>% select(timestamp, starts_with("axis"))
  agdb_10s %>%
    mutate(magnitude = sqrt(axis1^2 + axis2^2 + axis3^2))
  
  agdb_60s <- agdb_10s %>% collapse_epochs(60)

  sleep_ck <- agdb_60s %>%
    apply_cole_kripke() %>%
    apply_tudor_locke()
  write.csv(sleep_ck, file.path(output_folder_p, "sleep_times.csv"), row.names=FALSE)
  
  # Non-wear period detection with the Troiano and Choi algorithms
  non_wear <- agdb_60s %>% apply_troiano()#activity_threshold = 20, min_period_len = 40)
  write.csv(non_wear, file.path(output_folder_p, "actigraph_non_wear_times.csv"), row.names=FALSE)

}








