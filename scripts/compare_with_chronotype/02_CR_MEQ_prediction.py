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
This script predict MEQ with CR metrics acrophase. 

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
from matplotlib import pyplot as plt
import os
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from utils.read_file import load_config
from utils.error_handling import folderErrorHandling


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
output_coe_csv, output_res_csv, output_coe_ag_csv, output_res_ag_csv, fig_folder_path = (
    config.get(key, "") for key in [
        "Pred_coe_file", "Pred_res_file", "Pred_coe_file_ag", "Pred_res_file_ag", "fig_folder_path"
    ]
)

output_folder = os.path.join(root, CR_path)

folderErrorHandling(output_folder)
fig_folder = os.path.join(output_folder, fig_folder_path)
folderErrorHandling(fig_folder)
fig_file_path = os.path.join(fig_folder, "MEQ_prediction.png")


# Strings
str_Acti, str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV2, str_HRV3, str_HRV4 = (
    config.get(key, "") for key in [
        "str_Acti", "str_Watch",  "str_CBT", "str_SkinT", "str_HR", "str_HRV1", "str_HRV2", "str_HRV3", "str_HRV4"
    ]
)


def lr_calculation(X, y, var, lr_coe_whole, lr_res_whole):
    """Calculate linear regression model and return correlation and model metrics."""
    X_cons = sm.add_constant(X)
    mod = sm.OLS(y, X_cons)
    results = mod.fit()
    y_pred = results.predict(X_cons)
    #print(res.summary())

    coef_with_ci = results.params.round(2)
    ci = results.conf_int() 
    lr_coe_names = ["Variable", "Coefficients [95% CI] (VIF)"]
    
    for i, coef in enumerate(coef_with_ci):
        if i == 0: # constant is not recorded
            pass
        else:
            ci_lower, ci_upper = round(ci.iloc[i, 0], 2), round(ci.iloc[i, 1], 2)   
            # Single regression model
            if type(var) == str:
                lr_coe = [f"{var}", f"{coef}[{ci_lower},{ci_upper}]"]
            # Multiple regression model
            else:    
                vif = [variance_inflation_factor(X_cons, i) for i in range(X_cons.shape[1])]
                lr_coe = [f"{var}:{var[i-1]}", f"{coef}[{ci_lower},{ci_upper}], {round(vif[i], 2)}"]

            df_lr_coe = pd.DataFrame([lr_coe], columns=lr_coe_names)
            lr_coe_whole = pd.concat([lr_coe_whole, df_lr_coe], ignore_index=True)
            
    lr_res_names = ["Variable", "R-squared", "Adjusted R-squared", "F-statistic"]
    lr_res = [var, "{:.2f}".format(results.rsquared), "{:.2f}".format(results.rsquared_adj), "{:.2f}".format(results.fvalue)]
    df_lr_res = pd.DataFrame([lr_res], columns=lr_res_names)
    lr_res_whole = pd.concat([lr_res_whole, df_lr_res], ignore_index=True)
    
    return lr_coe_whole, lr_res_whole, y_pred



def corr_diag_plot(df):
    """Correlation plot between MEQ predictions and acrophase."""
    plt.style.use('seaborn-v0_8-colorblind')
    g = sns.PairGrid(df)
    g.fig.set_size_inches(6, 4)  # Adjust the width and height as needed
    
    g.map_lower(plt.scatter, s=0.5)
    g.map_lower(sns.regplot, scatter_kws={'s': 0.5}, line_kws={'color': 'blue', 'alpha': 0.3, 'linewidth':1})
    
    # Map diagonal plots with print_column_names
    g.map_diag(print_column_names)
    #g.map_diag(sns.histplot, kde=True)
    g.map_upper(corrfunc, cmap=plt.get_cmap('RdBu'))#, norm=plt.Normalize(vmin=-.5, vmax=.5))
    
    
    for ax in g.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
        
        if ax.get_ylabel() == "MEQ Predictions":
            ax.set_ylabel("MEQ\nPredictions", fontsize=8)
        else:
            ax.set_ylabel(ax.get_ylabel(), fontsize=8)
    
        ax.get_yaxis().set_label_coords(-0.35,0.5)
        ax.get_xaxis().set_label_coords(0.5,-0.4)
    
        ax.tick_params(labelsize=8, width = 0.4, length = 1) 
        plt.setp(ax.spines.values(), linewidth=0.5)
        
    g.fig.subplots_adjust(wspace=0.06, hspace=0.06) # equal spacing in both directions
    
    for i, ax in enumerate(g.axes.flatten()):
        if i == 0:
            ax.set_yticks([])
        elif i == 48:
            ax.set_xticks([])
    
    cbar_ax = g.fig.add_axes([1.003, 0.1, 0.015, 0.8])  # Adjust the position and size as needed
    scaled = 1
    colormap = plt.cm.get_cmap('RdBu') # 'plasma' or 'viridis'
    colors = colormap(scaled) 
    smp = plt.cm.ScalarMappable(cmap=colormap)
    smp.set_clim(vmin=-1.0, vmax=1.0)
    cbar = plt.colorbar(cax=cbar_ax, orientation='vertical', mappable=smp)
    cbar.outline.set_visible(False)
    cbar.set_label('Correlation Coefficient', fontsize=8)
    cbar.ax.tick_params(labelsize=8, width = 0.4, length = 1)   # set your label size here
    
    plt.show()
    g.savefig(os.path.join(fig_file_path), dpi=300)
    

def corrfunc(x, y, **kwds):
    """Add correlation with its significance level into the figure."""
    cmap = kwds['cmap']
    #norm = kwds['norm']
    ax = plt.gca()
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
    r, pvalue = pearsonr(x, y)
    facecolor = cmap((r+1)/2)
    #facecolor = cmap(norm(r))
    num=0
    if pvalue < 0.001:
        num = 3
    elif pvalue < 0.01:
        num = 2
    elif pvalue < 0.05:
        num = 1

    ax.set_facecolor(facecolor)
    lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
    ax.annotate(f"r={r:.2f}"+"*"*int(num), xy=(.5, .5), xycoords=ax.transAxes,
                color='white' if lightness < 0.7 else 'black', size=8, ha='center', va='center')

def print_column_names(x, **kwargs):
    """Print column names into the figure."""
    ax = plt.gca()
    name = x.name
    lines = name.split(' ')  # Split on underscores, adjust as needed
    
    if len(lines)>1:
        for i, line in enumerate(lines):
            y_position = 1- 0.33 * (i+1)
            ax.text(0.5, y_position, line, transform=ax.transAxes, fontsize=8.5, ha='center', va='center', weight='bold')
    else:
        ax.text(0.5, 0.5, x.name, transform=ax.transAxes, fontsize=8.5, ha='center', va='center', weight='bold')




def main():
    ref_name = 'MEQ'
    metrics_focus = 'time' # Acrophase representation in 24 h
    sensors_focus = [str_Acti, str_Watch, str_CBT, str_SkinT, str_HR, str_HRV1, str_HRV3]
    
    meq_score = pd.read_csv(os.path.join(input_q_folder, MEQ_score_file))
    cos_model = pd.read_csv(os.path.join(input_CR_folder, CR_model_file))
    age_gender = pd.read_csv(os.path.join(input_q_folder, age_gender_file))
    
    k_min, k_max = 2, 6 # Minimal and maximum of the multiple feature number
    
    # Age and gender as control variables, encode gender as dummy variable
    label_encoder = LabelEncoder()
    age_gender['Gender'] = label_encoder.fit_transform(age_gender['Gender'])
    age_gender = age_gender[['Age', 'Gender']]
    
    ## Merge MEQ scores and CR metrics
    data = {ref_name: meq_score[ref_name]}
    for which_sensor in sensors_focus:
        metrics_chosen = cos_model.loc[cos_model['test'] == which_sensor, metrics_focus].reset_index(drop=True)
        data[which_sensor] = metrics_chosen
        
    data = pd.DataFrame(data)
    
    ### Seperate features and ground truth
    all_features = data.drop(columns=[ref_name])
    multiple_features = data.drop(columns=[ref_name, str_Acti])
    y = data[ref_name]
    
    lr_res_whole = pd.DataFrame()
    lr_coe_whole = pd.DataFrame()
    
    lr_res_ag = pd.DataFrame()
    lr_coe_ag = pd.DataFrame()
    
    # Single linear regression model
    for var in all_features:
        X = all_features[var].values.reshape(-1, 1)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
        lr_coe_whole, lr_res_whole, _ = lr_calculation(X, y, var, lr_coe_whole, lr_res_whole)
        
        # Add age and gender as control variable
        X_ag = pd.concat([pd.DataFrame(X, columns=[var]), age_gender], axis=1).to_numpy()
        new_column = [var, "Age", "Gender"]
        lr_coe_ag, lr_res_ag, _ = lr_calculation(X_ag, y, new_column, lr_coe_ag, lr_res_ag)
    
    # Multiple linear regression model
    for num_fea in range(k_min, k_max+1):
        X = multiple_features
        
        k_best = SelectKBest(score_func=f_regression, k=num_fea)  # Select the top k features 
        X_topk = k_best.fit_transform(X, y)
        selected_feature_names = X.columns[k_best.get_support()]
    
        var = selected_feature_names.to_list()
        lr_coe_whole, lr_res_whole, y_pred = lr_calculation(X_topk, y, var,lr_coe_whole, lr_res_whole)
        
        # Add age and gender as control variable
        X_ag = pd.concat([pd.DataFrame(X_topk), age_gender], axis=1).to_numpy()
        new_column = var + ["Age", "Gender"]
        lr_coe_ag, lr_res_ag, _ = lr_calculation(X_ag, y, new_column, lr_coe_ag, lr_res_ag)
        
        # Plot prediction for the last multiple regression model
        if num_fea == k_max:
            df_viz = pd.DataFrame(X_topk, columns=selected_feature_names)
            df_viz["Predictions"] = y_pred
            corr_diag_plot(df_viz)
    
    lr_coe_whole.to_csv(os.path.join(output_folder, output_coe_csv), index=False)
    lr_res_whole.to_csv(os.path.join(output_folder, output_res_csv), index=False)
    
    lr_coe_ag.to_csv(os.path.join(output_folder, output_coe_ag_csv), index=False)
    lr_res_ag.to_csv(os.path.join(output_folder, output_res_ag_csv), index=False)
    

if __name__ == "__main__":
    main()
