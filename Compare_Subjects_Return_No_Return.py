import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split
from pcntoolkit.normative import estimate, evaluate
from Load_Genz_Data import load_genz_data
from plot_num_subjs import plot_num_subjs
from Utility_Functions import create_design_matrix, plot_data_with_spline, create_dummy_design_matrix
from Utility_Functions import barplot_performance_values, plot_y_v_yhat, makenewdir, movefiles
from Utility_Functions import write_ages_to_file
from scipy.stats import shapiro, levene, ttest_ind
from statsmodels.stats.multitest import multipletests


struct_var = 'cortthick'

working_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'

#load visit 1 (pre-COVID) data
visit=1
brain_goodv1, all_datav1, roi_ids = load_genz_data(struct_var, visit, working_dir)

#remove subject 525 who has an incidental finding
all_datav1 = all_datav1[~all_datav1['participant_id'].isin([525])]

# load all brain and behavior data for visit 2
visit = 2
brain_goodv2, all_datav2, roi_ids = load_genz_data(struct_var, visit, working_dir)

#remove subject 525 who has an incidental finding
all_datav2 = all_datav2[~all_datav2['participant_id'].isin([525])]

subjects_v1 = all_datav1['participant_id'].to_list()
subjects_v2 = all_datav2['participant_id'].to_list()

subjects_no_return = [s for s in subjects_v1 if s not in subjects_v2]

# Make dataframe containing only data for subjects that returned for visit 2
all_data_v1_yes_return = all_datav1[~all_datav1['participant_id'].isin(subjects_no_return)]
all_data_v1_yes_return.reset_index(inplace=True, drop=True)

# Make a dataframe containing only data for subjects that did not return for visit 2
all_data_v1_no_return = all_datav1[all_datav1['participant_id'].isin(subjects_no_return)]
all_data_v1_no_return.reset_index(inplace=True, drop=True)

# Test if ages of subjects in sample at time 1 were normal for subjects that did and subjects that did not return
stat_yes_return, p_yes_return = shapiro(all_data_v1_yes_return['age'])
stat_no_return, p_no_return = shapiro(all_data_v1_no_return['age'])

# They are not normally distributed, we aimed to have the same number of subjects at all ages
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.hist(all_data_v1_yes_return['age'], bins=list(range(9, 19)), align='left')
ax1.set_title('Age of Subjects That Returned')
ax1.set_xticks([9,11,13,15,17])
ax2.hist(all_data_v1_no_return['age'], bins=list(range(9, 19)), align='left')
ax2.set_title('Age of Subjects That Did Not Return')
ax2.set_xticks([9,11,13,15,17])
ax1.set_xlabel('Age of Subjects (years)')
ax2.set_xlabel('Age of Subjects (years)')
ax1.set_ylabel('Number of Subjects')
ax2.set_ylabel('Number of Subjects')
# plt.show(block=False)

# Test if gender of subjects in sample at time 1 were normal for subjects that did and subjects that did not return
stat_yes_return, p_yes_return = shapiro(all_data_v1_yes_return['sex'])
stat_no_return, p_no_return = shapiro(all_data_v1_no_return['sex'])

# They are not normally distributed, we aimed to have the same number of subjects at all ages
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.hist(all_data_v1_yes_return['sex'], bins=list(np.linspace(0,4, 8)))
ax1.set_title('Sex of Subjects That Returned')
ax1.set_xticks(list(range(1,3)), labels=['male', 'female'])
ax2.hist(all_data_v1_no_return['sex'], bins=list(np.linspace(0,4, 8)))
ax1.set_xlim(0, 2.8)
ax2.set_xlim(0, 2.8)
ax2.set_title('Sex of Subjects That Did Not Return')
ax2.set_xticks(list(range(1,3)), labels=['male', 'female'])
ax2.set_yticks(list(range(1, 20, 3)))
ax1.set_ylabel('Number of Subjects')
ax2.set_ylabel('Number of Subjects')
plt.tight_layout()
plt.show()

# Calculate average cortical thickiness by age across all regions for subjects who did and did not return
age_return = all_data_v1_yes_return['age']
all_data_v1_yes_return = all_data_v1_yes_return.drop(columns=['participant_id', 'age', 'sex', 'agedays'])
all_data_v1_yes_return = all_data_v1_yes_return.mean(axis=1)
all_data_v1_yes_return = pd.DataFrame(all_data_v1_yes_return, columns=['avgcortthick'])
all_data_v1_yes_return['age'] = age_return

age_no_return = all_data_v1_no_return['age']
all_data_v1_no_return = all_data_v1_no_return.drop(columns=['participant_id', 'age', 'sex', 'agedays'])
all_data_v1_no_return = all_data_v1_no_return.mean(axis=1)
all_data_v1_no_return = pd.DataFrame(all_data_v1_no_return, columns=['avgcortthick'])
all_data_v1_no_return['age'] = age_no_return

# Test if average cortical thickness of subjects in sample at time 1 were normal for subjects that did and did not return
# and check if they have equal variance
p_shapiro_yes_return_avgct_by_age = []
p_shapiro_no_return_avgct_by_age = []
p_levene_equal_var_by_age = []
p_ttest_equal_mean_by_age = []

# Create figure for plotting distributions of average cortical thickness for each age
fig, ax = plt.subplots(1,5, figsize=(16,8))
for i, age in enumerate(range(9, 19, 2)):
    df_yes = all_data_v1_yes_return.loc[all_data_v1_yes_return['age']==age,'avgcortthick'].tolist()
    stat_yes_return, p_shapiro_yes_return = shapiro(df_yes)
    p_shapiro_yes_return_avgct_by_age.append(p_shapiro_yes_return)
    df_no = all_data_v1_no_return.loc[all_data_v1_no_return['age'] == age, 'avgcortthick'].tolist()
    stat_no_return, p_shapiro_no_return =shapiro(df_no)
    p_shapiro_no_return_avgct_by_age.append(p_shapiro_no_return)
    levene_res = levene(df_yes, df_no)
    p_levene_equal_var_by_age.append(levene_res.pvalue)

    # Perform two-sample t-test
    t_statistic, p_ttest = ttest_ind(df_yes, df_no)
    p_ttest_equal_mean_by_age.append(p_ttest)

    # Plot histograms of cortical thickness data

    ax[i].hist(df_yes, bins=5, label='yes')
    ax[i].hist(df_no, bins=5, label='no')
    ax[i].set_title(f'{age} years\nAverage Cortical Thickness\np={p_ttest:.2f}')
    ax[i].set_xlabel('Average Cortical Thickness')
    ax[i].set_ylabel('Number of Subjects')
    ax[i].legend(title='returned')
plt.tight_layout()
plt.show()
mystop=1

reject, pcorr_shapiro_no_return, mas, mab = multipletests(p_shapiro_no_return_avgct_by_age, alpha=0.05, method='fdr_bh')
mystop=1