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
This script include several functions to visualize.
"""

import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.stats import pearsonr
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import matplotlib.dates as mdates


def plot_style():
    """Transparent background"""
    fig, ax = plt.subplots()
    ax.set_facecolor('none') 
    fig.set_size_inches(4, 3)
    return fig, ax

def stas_diff(y1, y2):
    """Calculate difference of y1 and y2"""
    mae = np.mean(np.abs(y1-y2))
    rmse = np.sqrt(np.mean((y1-y2) ** 2))
    t_stat, p_t = stats.ttest_ind(y1, y2)
    return round(mae, 4), round(rmse,4), round(t_stat, 4), round(p_t, 4)
            
def bland_altman_AC(data, plot_save=False, fig_folder=None):
    """Bland altman plot given difference and average of y1 and y2"""
    diff = data['diff']
    average = data['average']
    
    mean_diff = np.mean(diff)    
    std_diff = np.std(diff, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    #print("Agreement metrics: Mean Difference: ", mean_diff)
    #print("Limits of agreement ", loa_upper, loa_lower)
    
    fig, ax = plot_style()
    
    plt.scatter(average, diff, s=0.5, alpha=0.9)
    plt.axhline(mean_diff, color='gray', linestyle='solid', label='Mean Difference')
    plt.axhline(loa_upper, color='gray', linestyle='--', label='1.96 SD')
    plt.axhline(loa_lower, color='gray', linestyle='--')
    
    plt.text(0.8, 0.5, "Mean: "+str(round(mean_diff, 2)), fontsize=8, va='center', ha='center', transform=ax.transAxes)        
    plt.text(0.8, 0.75, "+1.96SD: "+str(round(loa_upper, 2)), fontsize=8, va='center', ha='center',transform=ax.transAxes)       
    plt.text(0.8, 0.3, "-1.96SD: "+str(round(loa_lower, 2)), fontsize=8, va='center', ha='center', transform=ax.transAxes)       

    plt.xlabel('Average of WatchAC and ActiAC', fontsize = 8)
    plt.ylabel('Difference (WatchAC - ActiAC)', fontsize = 8)
    plt.title('Bland-Altman Plot')
    
    pmi_x = -5
    pma_x = average.max() * 0.65
    pmi_y = diff.min() * 0.3
    pma_y = diff.max() * 0.3

    plt.xlim(pmi_x, pma_x)
    plt.ylim(pmi_y, pma_y)
    
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7, rotation=90, ha='right')
    plt.legend(fontsize = 7)
    plt.tight_layout()
    plt.show()

    if plot_save: 
        fig.savefig(os.path.join(fig_folder, 'Bland_Altman_AC.png'), dpi=300)
        
    return round(mean_diff, 4), round(loa_lower, 4), round(loa_upper, 4)


def regression_AC(y1, y2, plot_save=False, fig_folder=None):
    """Regression plot given y1 and y2"""

    fig, ax = plot_style()

    ax.scatter(y1, y2, label='ACs', s = 0.5)    
    ax.set_title('Regression between WatchAC and ActiAC')#'[' + patient_ID +']')

    model = LinearRegression()
    X = y1.values.reshape(-1, 1)
    y = y2.values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    #print(model.score(X, y))
    #print(model.coef_, model.intercept_)
    
    corr_coeff, p_corr = pearsonr(y1, y2)
    
    plt.plot(y1, y_pred, color="#696969", lw=1)
    plt.text(0.8, 0.5, r'$R^2$' + ' = ' +str(round(model.score(X, y), 2)), fontsize=8, va='center', ha='left', transform=ax.transAxes)        
    plt.text(0.8, 0.4, r'$y$' + ' = ' +str(round(model.coef_[0], 2)) +' * x +' +str(round(model.intercept_, 2)), fontsize=8, va='center', ha='left', transform=ax.transAxes)        
    plt.text(0.8, 0.3, r'$r$' + ' = ' +str(round(corr_coeff, 4)), fontsize=8, va='center', ha='left', transform=ax.transAxes)        
    plt.text(0.8, 0.2, r'$p$' + ' = ' +str(round(p_corr, 2)), fontsize=8, va='center', ha='left', transform=ax.transAxes)        

    plt.xlim(0, y1.max() * 0.8)
    plt.ylim(0, y2.max() * 0.8)
    
    #plt.xlim(-100, 15000)
    #plt.ylim(-100, 15000)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7, rotation=90, ha='right')
    
    plt.xlabel("WatchAC [AC]", fontsize=8)
    plt.ylabel('ActiAC [AC]', fontsize=8)
    plt.tight_layout()
    plt.show()

    if plot_save: 
        fig.savefig(os.path.join(fig_folder, 'Regression_AC.png'), dpi=300)

    return  round(corr_coeff,4), round(p_corr, 4) 



def scatter(x, y1, y2, label1, label2, plot_save=False, fig_folder=None, ylim_low = None):    
    """Scatter plot given y1 and y2 with regard to time"""

    fig, ax = plot_style()

    ax.scatter(x, y1, label=label1, s=0.5)
    ax.scatter(x, y2, label=label2, s=0.5)
    ax.set_title(label1 + 'and' + label2)
    
    plt.xlabel("Time", fontsize=8)
    plt.ylabel(label1 + 'and' + label2, fontsize=8)
    ax.legend(fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7, rotation=90, ha='right')
    
    # Text in the x-axis will be displayed in 'YYYY-mm' format.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    if ylim_low is not None:
        plt.ylim(min(y1.min(), y2.min()), max(y1.max(), y2.max()))
    else:
        plt.ylim(0, max(y1.max(), y2.max()))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    if plot_save: 
        fig.savefig(os.path.join(fig_folder, 'Scatter_'+label1+'_'+label2+'.png'), dpi=300)
        

