# Circadian Rhythm Metrics based on Activity, body temperature and Heart rate (CHARM)

## Introduction
This python package is to calculate circadian rhythm metrics from sensors which has been wore for 14 days. 

## Configuration and Usage

### 1. Clone this repository.

### 2. Create two python virtual environments.

### 3. Activite each environment and install the required packages seperately.
    In the first environment: pip install -r requirements1.txt
    In the second environment: pip install -r requirements2.txt
    
### 4. In the specific environment, run the script.
For example, python AC_comparison.py. Make sure you have all required input files before you run the script.

#### Main Module1: Data processing (Run under the 1st python environment)
    1. Actigraph_AC.py: obtain Activity counts obtained from Actigraph (ActiLife software)
    2. AC_comparison.py: compare Activity counts calculated from Samsung Watch 5 with Activity counts obtained from Actigraph.
    3. Core_temperature.py: obtain core body temperature obtained from GreenTeg CORE sensor.
    4. HR_calculation.py: obtain heart rate obtained from Samsung Watch 5 and calculate 4 heart rate variability (HRV) metrics.
    5. Miss_stats.py: Calculate data missing rate of Actigraph, Watch and CORE sensor.
    6. Questionnaire.py: Calculate demographic information and MEQ questionnaire scores.
    
#### Main Module2: Circadian calculation and comparison (Run under the 2nd python environment)
    1. Cosinor_metrics.py: Calculate circadian rhythm metrics, including amplitude, acrophase, and mesor.
    2. CR_non_parametric.py: Calculate non-parametrics metrics, including IS, IV, L5, M10, RA.
    3. Cosinor_ID_comparison.py: Compare circdian rhythm metrics between two individuals.
    4. CR_comparison.py: Using the metrics calculated from Actigraph as the reference, compare the metrics calculated from other sensor data.

#### Main Module3: Compare with chronotype (Run under the 2nd python environment)
    1. CR_MEQ_correlation.py: Calculate correalation between acrophase from different sensor type with MEQ questionnaire.
    2. CR_MEQ_prediction.py: Prediction model for MEQ questionnaire based on acrophase from different sensor type.
    3. CR_PCA_circle.py: Biplot to illustrate the relationship between acrophase and MEQ questionnaire.
    4. CR_group_comparison.py: Acrophase comparison between different chronotype group.

#### Supporting Modules: 
##### Modules Utils:
    1. error_handling.py
    2. read_file.py
    3. visulization.py

##### Modules algorithms:
    1. ActivityCounts: class to calculate activity counts and compare.
    2. counts_cal_.py

##### Modules time_preprocessing:
    1. Watch_charging.py: obtains time fragments after excluding smartwatch charging time.
    2. Watch_times.py: obtains smartwatch valid wearing time for acceleration and heart rate. It excludes the time when smartwatch was charging or when no related file was not recorded.
    3. Actigraph_times.R: obtains Actigraph non-wearing and sleep time.

##### Modules others:
    1. Repeated_measures_corr.py: calculate repeated measures correlation betweeen Activity Counts of Actigraph and Watch. To run this script, need to create a seperate environment and install package Pingouin.

## Pipeline

<img src="https://github.com/user-attachments/assets/3cd077ce-b28a-4193-be44-ea5a57617824" alt="pipe" width="800"/>
<br>

## Developer

Fan Wu, ETH Zurich, 2024

## Citation
Please cite our paper:

Wu, F., Langer, P., Shim, J., Fleisch, E. and Barata, F., 2024. Comparative efficacy of commercial wearables for circadian rhythm home monitoring from activity, heart rate, and core body temperature. IEEE Journal of Biomedical and Health Informatics.

## Data

🔥 Please reach out to the corresponding author to sign the necessary data consent form in order to obtain the data. Once you have the data, place the data folder at the same level as the 'script' folder.

Here is the structure of the data.

```bash
sample_data
├── 01 (participant)
│   ├── Actigraph
│   │   └── 01.csv (Actigraph Activity count file)
│   ├── Core
│   │   └── 01.csv (Core temperature data file)
│   └── Samsung
│       └── Smartwatch
│           ├── AccelerometerData
│           │   └── 25.05.23_00 (Raw acceleration file from the smartwatch for one hour)
│           ├── BatteryData
│           └── HeartRateData
├── 02
│   └── .... (Other participant data)
questionnaire
├── Baseline.csv (demographic information)
├── Q1.csv (MEQ questionnaire at the first time)
├── Q2.csv 
└── Q3.csv 
output
└── Time (So no need to run Modules time_preprocessing)
    └── 01 (participant)
        ├── actigraph_non_wear_times.csv
        ├── battery_miss.csv
        ├── hr_times.csv
        ├── sleep_times.csv
        └── watch_acc_times.csv
times.csv (start point and end point of each participant)

```



<img src="https://github.com/ADAMMA-CDHI-ETH-Zurich/CROCOanalysis/assets/44665480/e985c7d8-215c-4444-b8c2-f2b97c615c28" alt="eth" width="200"/>

<img src="https://github.com/ADAMMA-CDHI-ETH-Zurich/CROCOanalysis/assets/44665480/88d6a90b-11a0-4c71-90a0-e02bf58cfa57" alt="cdhi" width="250"/>

