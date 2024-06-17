####
# This program calculate the mean Z score and confidence intervals for males and females. This is one estimate
# of effect size.
####

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

working_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'

# Specify filename for post-covid z-scores
Z_time2_file = f'{working_dir}/predict_files/cortthick/Z_scores_by_region_postcovid_testset_Final.txt'

# Load Z scores from post-covid data
Z2 = pd.read_csv(Z_time2_file)

# Add sex to dataframe
subject_id = Z2['participant_id'].tolist()

# Calculate sex for each subject
sex_val = [0 if s % 2 == 0 else 1 for s in subject_id]

# Add to dataframe
Z2['sex'] = sex_val

# Make separate dataframes for males and females
Z2_female = Z2[Z2['sex']==0]
Z2_male = Z2[Z2['sex']==1]

# Create list of brain regions
rois = Z2.columns.values.tolist()
rois.remove('participant_id')
rois.remove('sex')

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
axs[0].errorbar(x=range(len(mean_male)), y=mean_male, yerr=[mean_male - lower_ci_male,
                                                upper_ci_male - mean_male], fmt='o', label='Male', color='blue', markersize=3)

# Plotting mean values with error bars for females
axs[1].errorbar(x=range(len(mean_female)), y=mean_female, yerr=[mean_female - lower_ci_female,
                                                upper_ci_female - mean_female], fmt='o', label='Female', color='green', markersize=3)
for ax in [0, 1]:
    axs[ax].set_ylabel('Mean Z-score', fontsize=12)
    if ax == 0:
        gender = 'Males'
    else:
        gender = 'Females'
    axs[ax].set_title(f'{gender}: Mean Z-score with Confidence Intervals by Brain Region ')

    axs[ax].set_xticks(range(len(mean_female)), mean_female.index, rotation=90, fontsize=11)
    axs[ax].set_xlim(-0.8, len(mean_female) - 0.5)
    axs[ax].set_ylim(-1.8, 0.9)
    axs[ax].axhline(y=0, linestyle='--', color='gray')
    axs[ax].tick_params(axis='y', labelsize=10)

plt.savefig(f'{working_dir}/Mean Z-score with Confidence Intervals for both genders.png')
plt.show()


