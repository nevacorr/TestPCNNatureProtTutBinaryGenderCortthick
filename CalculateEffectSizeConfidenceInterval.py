#####
# This program uses bootstrapping to calculate confidence interval for effect size of cortical thickness deviations
# in post-covid data as compared to what woudl be expected from the pre-covid data.
######
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Load_Genz_Data import load_genz_data
from Utility_Functions import makenewdir
from Utility_Functions import read_ages_from_file
from sklearn.utils import resample
from calculate_ct_from_model import apply_normmodel_postcovid

struct_var = 'cortthick'
show_nsubject_plots = 0
show_plots = 0
spline_order = 1
spline_knots = 2
working_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
n_bootstrap = 2
alpha = 0.05

######################## Predict Cortical Thickness from to Resampled Post-Covid Data ############################

# Load all brain and behavior data for visit 2
visit = 2
brain_good, all_data, roi_ids = load_genz_data(struct_var, visit, working_dir)

# Load brain and behavior data for visit 1
visit = 1
brain_v1, all_v1, roi_v1 = load_genz_data(struct_var, visit, working_dir)

# Extract subject numbers from visit 1 and find subjects in visit 2 that aren't in visit 1
subjects_visit1 = all_v1['participant_id']
rows_in_v2_but_not_v1 = all_data[~all_data['participant_id'].isin(all_v1['participant_id'])].dropna()
subjs_in_v2_not_v1 = rows_in_v2_but_not_v1['participant_id'].copy()
subjs_in_v2_not_v1 = subjs_in_v2_not_v1.astype(int)
# Only keep subjects at 12, 14 and 16 years of age (subject numbers <400) because cannot model 18 and 20 year olds
subjs_in_v2_not_v1 = subjs_in_v2_not_v1[subjs_in_v2_not_v1 < 400]

# Make file diretories for output
makenewdir('predict_files_bootstrap/')
makenewdir('predict_files_bootstrap/{}'.format(struct_var))
makenewdir('predict_files_bootstrap/{}/plots'.format(struct_var))
makenewdir('predict_files_bootstrap/{}/ROI_models'.format(struct_var))
makenewdir('predict_files_bootstrap/{}/covariate_files'.format(struct_var))
makenewdir('predict_files_bootstrap/{}/response_files'.format(struct_var))

# Only include subjects that were not in the training set
fname='{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(working_dir, struct_var)
subjects_to_include = pd.read_csv(fname, header=None)
subjects_to_include = pd.concat([subjects_to_include, subjs_in_v2_not_v1])
brain_good = brain_good[brain_good['participant_id'].isin(subjects_to_include[0])]
all_data = all_data[all_data['participant_id'].isin(subjects_to_include[0])]

# Reset indices
brain_good.reset_index(inplace=True)
all_data.reset_index(inplace=True, drop=True)
#read agemin and agemax from file
agemin, agemax = read_ages_from_file(struct_var)

# Replace gender with binary gender
all_data.loc[all_data['sex']==2, 'sex'] = 0

# Initialize DataFrames to store effect size for each gender for all bootstrap samples
effect_size_by_region_final_f = pd.DataFrame()
effect_size_by_region_final_m = pd.DataFrame()

# specify paths
training_dir = '{}/data/{}/ROI_models/'.format(working_dir, struct_var)
out_dir = '{}/predict_files_bootstrap/{}/ROI_models/'.format(working_dir, struct_var)
#  this path is where ROI_models folders are located
predict_files_dir = '{}/predict_files_bootstrap/{}/ROI_models/'.format(working_dir, struct_var)

for i in range(n_bootstrap):
    print(f'BOOTSTRAP {i} out of {n_bootstrap}')

    all_data = resample(all_data, stratify=all_data[['age', 'sex']])
    all_data.reset_index(drop=True, inplace=True)
    all_data.sort_values(by='participant_id', inplace=True)
    mystop=0
    effect_size_by_region = (
        apply_normmodel_postcovid(all_data, roi_ids, working_dir, struct_var, agemin, agemax, spline_order,
                                  spline_knots, training_dir, out_dir, predict_files_dir))

    effect_size_by_region_final_f = pd.concat([effect_size_by_region_final_f, effect_size_by_region.loc[0.0, :]], axis=1)
    effect_size_by_region_final_m = pd.concat([effect_size_by_region_final_m, effect_size_by_region.loc[1.0,:]], axis=1)

# Save effect sizes to file
effect_size_by_region_final_f.to_csv(f'{working_dir}/tmp_effect_f.csv')
effect_size_by_region_final_m.to_csv(f'{working_dir}/tmp_effect_m.csv')

# Load effect sizes from file
effect_size_by_region_final_f = pd.read_csv(f'{working_dir}/tmp_effect_f.csv', index_col=0)
effect_size_by_region_final_m = pd.read_csv(f'{working_dir}/tmp_effect_m.csv', index_col=0)

# Get list of region names
roi_ids = effect_size_by_region_final_f.index.values.tolist()

# Sort effect sizes across all bootstrap samples for females and males
# Females
data_array = effect_size_by_region_final_f.to_numpy()
sorted_data = np.sort(data_array, axis=1)
sorted_df_f = pd.DataFrame(sorted_data, index=effect_size_by_region_final_f.index)
# Males
data_array = effect_size_by_region_final_m.to_numpy()
sorted_data = np.sort(data_array, axis=1)
sorted_df_m = pd.DataFrame(sorted_data, index=effect_size_by_region_final_m.index)

# Create datafames to store mean effects and confidence intervals
CI_df = pd.DataFrame(index=roi_ids, columns=['lower_CI_f', 'upper_CI_f', 'lower_CI_m', 'upper_CI_m'])
mean_effect_df = pd.DataFrame(index=roi_ids, columns=['female', 'male'])

# For each ROI, calculate upper and lower confidence interval and mean effect for males and females
for roi in roi_ids:
    # Calculate lower and upper confidence intervals for males and females
    CI_df.loc[roi,'lower_CI_f'] = np.percentile(sorted_df_f.loc[roi,:], alpha / 2 * 100)
    CI_df.loc[roi,'upper_CI_f'] = np.percentile(sorted_df_f.loc[roi,:], 97.5)
    CI_df.loc[roi,'lower_CI_m'] = np.percentile(sorted_df_m.loc[roi,:], alpha / 2 * 100)
    CI_df.loc[roi,'upper_CI_m'] = np.percentile(sorted_df_m.loc[roi,:], 97.5)
    mean_effect_df.loc[roi, 'female'] = sorted_df_f.loc[roi,:].mean()
    mean_effect_df.loc[roi, 'male'] = sorted_df_m.loc[roi, :].mean()

# Create a figure to plot mean effect with confidence intervals for each brain region
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Create subplot for females
ax1.errorbar(range(len(mean_effect_df)), abs(mean_effect_df['female']),
              yerr=[abs(mean_effect_df['female'] - CI_df['lower_CI_f']), abs(mean_effect_df['female'] - CI_df['upper_CI_f'])],
              fmt='o', label='Female', color='green')
ax1.set_title('Female Effect Size')
ax1.set_xlabel('Brain Region')
ax1.set_ylabel('Mean Effect')
ax1.set_xticks(np.arange(len(roi_ids)))
ax1.set_xticklabels(roi_ids, rotation=45, ha='right')

# Create subplot for males
ax2.errorbar(range(len(mean_effect_df)), abs(mean_effect_df['male']),
              yerr=[abs(mean_effect_df['male'] - CI_df['lower_CI_m']), abs(mean_effect_df['male'] - CI_df['upper_CI_m'])],
              fmt='o', label='Male', color='blue')
ax2.set_title('Male Effect Size')
ax2.set_xlabel('Brain Region')
ax2.set_ylabel('Mean Effect')
ax2.set_xticks(np.arange(len(roi_ids)))
ax2.set_xticklabels(roi_ids, rotation=45, ha='right')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

mystop=1
