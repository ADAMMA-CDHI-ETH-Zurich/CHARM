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
This script calculates repeated measures correlation.
"""

import pandas as pd
import os
import sys
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils.read_file import load_config


############### Define global parameters ###############
# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)


# Input folder and files
root, input_path, AC_file = (
    config.get(key, "") for key in [
        "output_root", "sensor_folder", "AC_file"
    ]
)

input_folder = os.path.join(root, input_path)

str_time, str_Acti, str_Watch = (
    config.get(key, "") for key in [
        "str_time", "str_Acti", "str_Watch"
    ]
)

subfolders = [
    f for f in os.listdir(input_folder)
    if os.path.isdir(os.path.join(input_folder, f)) and f.isdigit()
]

patient_list = sorted(subfolders, key=lambda x: int(x))

df_rmc = pd.DataFrame()

for patient_ID in patient_list:
    print("="*30)
    print(patient_ID)   
    df = pd.read_csv(os.path.join(input_folder, patient_ID, AC_file))
    df['Subject'] = patient_ID
    df_rmc = pd.concat([df_rmc, df])

df_rmc = df_rmc.reset_index()
df_rmc = df_rmc[['Subject', str_Watch, str_Acti]]

rmc_result = pg.rm_corr(data=df_rmc, x=str_Watch, y=str_Acti, subject='Subject')
print(rmc_result)
print(str(round(rmc_result.values[0][0], 4)), str(round(rmc_result.values[0][2], 4)))


sns.set(style='darkgrid', font_scale=0.8)

rmc = pg.plot_rm_corr(data=df_rmc, x=str_Watch, y=str_Acti,
                    subject='Subject', legend=False,
                    kwargs_facetgrid=dict(height=4.5, aspect=1.5, palette='Spectral'),
                    kwargs_line=dict(linewidth=0.7),
                    kwargs_scatter=dict(alpha=1, s=0.3)
                    )

fig = rmc.fig.set_dpi(400) 
ax = rmc.ax
ax.text(0.7, 0.2, r'$r_m$' + ' = ' + str(round(rmc_result.values[0][0], 4)), ha='left', va='center', transform=ax.transAxes, fontsize=12)
ax.text(0.7, 0.15, r'$p$' + ' = ' + str(round(rmc_result.values[0][2], 4)), ha='left', va='center', transform=ax.transAxes, fontsize=12)

labels_patient = [format(num, '02') for num in range(1, 37)]
legend=plt.legend(title='Participant ID', labels=labels_patient, fontsize="8", ncol=2,  bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
legend.get_title().set_fontsize('10') #legend 'Title' fontsize

plt.xlabel("Samsung [AC]")
plt.ylabel("Actigraph [AC]")

plt.show()