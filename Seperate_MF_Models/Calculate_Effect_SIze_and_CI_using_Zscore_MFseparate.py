####
# This program calculate the mean Z score and confidence intervals for males and females. This is one estimate
# of effect size.
####

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

working_dir = os.getcwd()

# Specify filenames for post-covid z-scores
Z_female_time2_file = f'{working_dir}/predict_files/Z_time2_female.csv'
Z_male_time2_file = f'{working_dir}/predict_files/Z_time2_male.csv'

# Load Z scores from post-covid data for males and females
Z2_female = pd.read_csv(Z_female_time2_file)
Z2_male = pd.read_csv(Z_male_time2_file)

# Create list of brain regions
rois = Z2_female.columns.values.tolist()
rois.remove('participant_id')

# Calculate standard deviations for all brain regions for males and females
Z2_stats = pd.DataFrame(index=['mean_female', 'mean_male', 'std_female', 'std_male'])

for col in rois:
    Z2_stats.loc['mean_female', col] = np.mean(Z2_female.loc[:,col])
    Z2_stats.loc['mean_male', col] = np.mean(Z2_male.loc[:,col])
    Z2_stats.loc['std_female', col] = np.std(Z2_female.loc[:,col])
    Z2_stats.loc['std_male', col] = np.std(Z2_male.loc[:,col])

Z2_stats.loc['upper_CI_female',:] = (
        Z2_stats.loc['mean_female',:] + 1.96 * Z2_stats.loc['std_female'] / math.sqrt(Z2_female.shape[0] - 1))
Z2_stats.loc['lower_CI_female',:] = (
        Z2_stats.loc['mean_female',:] - 1.96 * Z2_stats.loc['std_female'] / math.sqrt(Z2_female.shape[0] - 1))
Z2_stats.loc['upper_CI_male',:] = (
        Z2_stats.loc['mean_male',:] + 1.96 * Z2_stats.loc['std_male'] / math.sqrt(Z2_male.shape[0] - 1))
Z2_stats.loc['lower_CI_male',:] = (
        Z2_stats.loc['mean_male',:] - 1.96 * Z2_stats.loc['std_male'] / math.sqrt(Z2_male.shape[0] - 1))

# Calculate effect size
cohensd_female = Z2_stats.loc['mean_female']
cohensd_male = Z2_stats.loc['mean_male']

# Count effect size values above or equal to 0.5

female_count_above_threshold = (cohensd_female <= -0.5).sum()
male_count_above_threshold = (cohensd_male <= -0.5).sum()

# Remove prefix from column names
Z2_stats.columns = Z2_stats.columns.str.replace('cortthick-', '')

# Extract mean values and confidence intervals
mean_female = Z2_stats.loc['mean_female']
mean_male = Z2_stats.loc['mean_male']
upper_ci_female = Z2_stats.loc['upper_CI_female']
lower_ci_female = Z2_stats.loc['lower_CI_female']
upper_ci_male = Z2_stats.loc['upper_CI_male']
lower_ci_male = Z2_stats.loc['lower_CI_male']

# Plotting
fig, axs = plt.subplots(2, figsize=(14, 18), constrained_layout=True)

# Plotting mean values with error bars for males
axs[1].errorbar(x=range(len(mean_male)), y=mean_male, yerr=[mean_male - lower_ci_male,
                                                upper_ci_male - mean_male], fmt='o', label='Males', color='blue', markersize=3)

# Plotting mean values with error bars for females
axs[0].errorbar(x=range(len(mean_female)), y=mean_female, yerr=[mean_female - lower_ci_female,
                                                upper_ci_female - mean_female], fmt='o', label='Females', color='crimson', markersize=3)

for ax in [0, 1]:
    axs[ax].set_ylabel('Mean Effect Size', fontsize=12)
    if ax == 1:
        gender = 'Males'
    else:
        gender = 'Females'
    # axs[ax].set_title(f'{gender}: Mean Z-score with Confidence Intervals by Brain Region')
    axs[ax].set_xticks(range(len(mean_female)), mean_female.index, rotation=90, fontsize=11)
    axs[ax].set_xlim(-0.8, len(mean_female) - 0.5)
    axs[ax].set_ylim(-1.8, 1.05)
    axs[ax].axhline(y=0, linestyle='--', color='gray')
    axs[ax].tick_params(axis='y', labelsize=10)
    axs[ax].legend(loc='upper left', fontsize=12)

plt.savefig(f'{working_dir}/Mean_Z-score_for_each_region_with_CI_M_F_sep.png')
plt.show()

# Plot effect size without confidence intervals
fig, axs =plt.subplots(2, constrained_layout=True, figsize=(14, 18),)
axs[1].plot(cohensd_male, marker='o', color='b', linestyle='None', label='Males')
axs[0].plot(cohensd_female, marker='o', color='crimson', linestyle='None',  label='Females')
for ax in [0, 1]:
    axs[ax].set_ylabel("Effect Size", fontsize=14)
    if ax == 1:
        gender = 'Males'
    else:
        gender = 'Females'
    # axs[ax].set_title(f"{gender}: Effect Size by Brain Region", fontsize=16)
    axs[ax].set_xticks(range(len(mean_female)), mean_female.index, rotation=90, fontsize=14)
    axs[ax].set_xlim(-0.8, len(mean_female) - 0.5)
    axs[ax].set_ylim(-1.4, 0.6)
    axs[ax].axhline(y=0.0, linestyle='--', color='gray')
    axs[ax].legend(loc = 'upper left', fontsize=12)
plt.savefig(f'{working_dir}/Effect Size for both genders no CI MF Separate Models.png')
plt.show()

