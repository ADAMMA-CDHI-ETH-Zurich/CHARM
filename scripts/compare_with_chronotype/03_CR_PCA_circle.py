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
This script draws PCA circle for the selected CR metrics for all sensor data. 

Input:
- input_folder: Path where CR metrics stores.
- cosinor_file: File path which stores cosinor metrics for all individuals.
- np_file: File path which stores non-parametric metrics for all individuals.

Output:
- output_folder: Folder path which stores PCA circle figure.
- fig_file_path: File path of the PCA circle.
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import numpy as np
from matplotlib.patches import Circle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap
from utils.read_file import load_config
from utils.error_handling import folderErrorHandling

############### Define global parameters ###############

# Load configuration and define variables
current_dir = os.path.dirname(os.path.abspath(__file__))
config = load_config(current_dir)

# Input folder and files
root, cr_path, cosinor_file, np_file = (
    config.get(key, "") for key in [
        "output_root", "CR_folder", "CR_model_file", "CR_non_para_file"
    ]
)

input_folder = os.path.join(root, cr_path)


# Output folder and files
fig_folder_path, = (
    config.get(key, "") for key in [
        "fig_folder_path",
    ]
)

output_folder = os.path.join(root, cr_path)

fig_folder = os.path.join(output_folder, fig_folder_path)
folderErrorHandling(fig_folder)

fig_file_path = os.path.join(fig_folder, "PCA.png")


# Strings
str_ID, str_HRV1, str_HRV2, str_HRV3, str_HRV4 = (
    config.get(key, "") for key in [
        "str_ID", "str_HRV1", "str_HRV2", "str_HRV3", "str_HRV4"
    ]
)
strs_not_relevant = [str_ID, str_HRV2, str_HRV4]



def pca_circle(pca_data, title):
    """Plot PCA circle."""
    plt.style.use('seaborn-v0_8-colorblind')

    X = pca_data.values
    Xstd = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    Xpca = pca.fit_transform(Xstd)
    
    labels = [var for i, var in enumerate(pca.explained_variance_ratio_ * 100)]
    #print(labels)
    print(labels[0])
    
    feature_names = list(pca_data)
    
    ccircle = []
    eucl_dist = []
    texts = []
    for i,j in enumerate(X .T):
        corr1 = np.corrcoef(j,Xpca[:,0])[0,1]
        corr2 = np.corrcoef(j,Xpca[:,1])[0,1]
        ccircle.append((corr1, corr2))
        eucl_dist.append(np.sqrt(corr1**2 + corr2**2))
        
    fig, axs = plt.subplots(figsize=(3.5, 3.5), dpi=300)
    
    for i,j in enumerate(eucl_dist):
        
        original_cmap = plt.cm.RdBu
        new_cmap = LinearSegmentedColormap.from_list('winter_half', original_cmap(range(int(original_cmap.N / 4 * 1), int(original_cmap.N / 4 * 3))))    
        arrow_col = new_cmap((eucl_dist[i] - np.array(eucl_dist).min())/\
                                (np.array(eucl_dist).max() - np.array(eucl_dist).min()) )
        axs.arrow(0,0, # Arrows start at the origin
                 ccircle[i][0],  #0 for PC1
                 ccircle[i][1],  #1 for PC2
                 lw = 2, # line width
                 length_includes_head=True, 
                 color = arrow_col,
                 fc = arrow_col,
                 head_width=0.05,
                 head_length=0.05)
        
        texts.append(axs.text(ccircle[i][0] * 1.05, ccircle[i][1] * 1.05, feature_names[i], fontsize=8, bbox=dict(facecolor='lightgray', edgecolor='none', boxstyle='round, pad=0.05')))
        #axs.text(ccircle[i][0]*1.15,ccircle[i][1]*1.15, feature_names[i], fontsize=10, bbox=dict(facecolor='lightgray', boxstyle='round, pad=0.1'))
        axs.xaxis.set_tick_params(width=0.3, length = 0.5)
        axs.yaxis.set_tick_params(width=0.3, length = 0.5)

    adjust_text(texts, ax=axs)#, only_move={'points':'x', 'texts':'x'})
    
    # Draw the unit circle, for clarity
    circle = Circle((0, 0), 1, facecolor='none', edgecolor=(0.5, 0.5, 0.5), linewidth=1, alpha=1)
    axs.add_patch(circle)
    plt.xlim([-1.05, 1.05])
    plt.ylim([-1.05, 1.05]) 
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    axs.set_xlabel(f"Principal component 1 ({labels[0]:.2f}%)",  fontsize=8)
    axs.set_ylabel(f"Principal component 2 ({labels[1]:.2f}%)",  fontsize=8)

    plt.box(False)

    axs.set_facecolor('none')  # Transparent background
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_color('black')
    axs.spines['left'].set_color('black')
    axs.spines['bottom'].set_linewidth(0.3)
    axs.spines['left'].set_linewidth(0.3)

    plt.title(title, fontsize=8)
    
    plt.tight_layout()
    
    plt.savefig(fig_file_path, dpi=300)

    plt.show()


def main():
    
    cos_model = pd.read_csv(os.path.join(input_folder, cosinor_file))
    np_model = pd.read_csv(os.path.join(input_folder, np_file))
    
    cr_metric_sd = cos_model[['ID', 'test','amplitude', 'time', 'mesor']]
    cr_metric_sd.rename(columns={'test': 'Measurement', 'time': 'Acrophase'}, inplace=True)
    complete_data = pd.merge(cr_metric_sd, np_model, on=['ID', 'Measurement'], how='inner')
    complete_data_metric = complete_data[['ID', 'Measurement', 'Acrophase']].pivot(index='ID', columns='Measurement', values='Acrophase')
    
    complete_data_metric = complete_data_metric.reset_index()

    complete_data_metric = complete_data_metric.drop(columns=strs_not_relevant)
    
    pca_circle(complete_data_metric, "Acrophase")
    
    #complete_data_sensor = complete_data[complete_data['Measurement'] == 'ActiAC']
    #pca_circle(complete_data_sensor, "Actigraph")

        
if __name__ == "__main__":
    main()