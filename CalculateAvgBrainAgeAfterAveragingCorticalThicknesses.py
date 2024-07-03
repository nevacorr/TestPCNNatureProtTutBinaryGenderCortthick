#####
# This program calculates brain age acceleration based on the adolescent data. It averages cortical thickness across
# all brain regions. It fits a model on the precovid data and evaluates the model on the post-covid data.
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utility_Functions import plot_age_acceleration
from scipy.stats import percentileofscore
from Load_Genz_Data import load_genz_data
from plot_num_subjs import plot_num_subjs
from calculate_avg_brain_age_acc_across_select_regions import calculate_avg_brain_age_acceleration_make_model
from calculate_avg_brain_age_acc_across_select_regions import calculate_avg_brain_age_acceleration_apply_model
from calculate_avg_brain_age_acceleration_apply_model_bootstrap import calculate_avg_brain_age_acceleration_apply_model_bootstrap

struct_var = 'cortthick'
show_plots = 0  #set to 1 to show training and test data y vs yhat and spline fit plots. Set to 0 to save to file.
show_nsubject_plots = 0 #set to 1 to show number of subjects in analysis
spline_order = 1
spline_knots = 2
perform_train_test_split_precovid = 1  # flag indicating whether training set was split into train and validation set
filepath = os.getcwd()
subjects_to_exclude = [525]  # subjects to exclude from analysis. Subject 525 had an incidental finding
calc_CI_age_acc_bootstrap = 0  # specify whether to run bootstrap analysis for CI calculation
nbootstrap = 1000  #specify number of bootstraps

# Turn off interactive mode, don't show plots unless plt.show() is specified
plt.ioff()

# Load visit 1 data
visit=1
brain_good, all_data, roi_ids = load_genz_data(struct_var, visit, filepath)

# Remove subject 525 who has an incidental finding
all_data = all_data[~all_data['participant_id'].isin(subjects_to_exclude)]

# Read in file of subjects in post-COVID test set
fname='{}/visit2_all_subjects_used_in_test_set_cortthick.txt'.format(filepath, struct_var)
subjects_test = pd.read_csv(fname, header=None)

# Exclude subjects at 9, 11 and 13 who are in test set from dataframe of visit 1 data
all_data = all_data[~all_data['participant_id'].isin(subjects_test[0])]

# Replace gender with gender=0 female gender =1 male
all_data.loc[all_data['sex']==2, 'sex'] = 0

# Plot number of subjects of each gender by age who are included in training data set
if show_nsubject_plots:
    plot_num_subjs(all_data, 'Subjects by Age with Pre-COVID Data\nUsed to Create Model\n'
                   '(Total N=' + str(all_data.shape[0]) + ')', struct_var, 'pre-covid_norm_model', filepath)

# Drop amy rows with any missing values
all_data = all_data.dropna()
all_data.reset_index(inplace=True, drop=True)

# If validation data was not used in model creation, exclude these subjects from dataframe
if perform_train_test_split_precovid == 1:
    fname_train = f'{filepath}/train_subjects_excludes_validation.csv'
    subjects_train = pd.read_csv(fname_train, header=None)
    subjects_train = subjects_train[0].tolist()
    all_data = all_data[all_data['participant_id'].isin(subjects_train)]

# Separate the brain features (response variables) and predictors (age, gender) in to separate dataframes
all_data_features_orig = all_data.loc[:,roi_ids]
all_data_covariates = all_data[['age', 'agedays', 'sex']]

# Average cortical thickness across all regions for each subject
all_data_features = all_data_features_orig.mean(axis=1).to_frame()
all_data_features.rename(columns={0:'avgcortthick'},  inplace=True)

# Create model for when cortthick is averaged across the entire brain
model_dir, agemin, agemax = calculate_avg_brain_age_acceleration_make_model('allreg',
                        all_data, all_data_covariates, all_data_features, struct_var, show_plots,
                        spline_order, spline_knots, filepath)

# Specify visit number
visit = 2
# Load all brain and behavior data for visit 2
brain_good, all_datav2, roi_ids = load_genz_data(struct_var, visit, filepath)

# Load test subject numbeers
fname = '{}/visit2_all_subjects_used_in_test_set_{}.txt'.format(filepath, struct_var)
my_file = open(fname, 'r')
test_subjects_txt = my_file.read()
test_subjects = test_subjects_txt.split("\n")
my_file.close()
while ("" in test_subjects):
    test_subjects.remove("")
test_subjects = [int(i) for i in test_subjects]

# Create a dataframe with just test subject data
all_datav2 = all_datav2[all_datav2['participant_id'].isin(test_subjects)]

# Replace gender with binary gender
all_datav2.loc[all_datav2['sex'] == 2, 'sex'] = 0

# Show number of subjects by gender and age
if show_nsubject_plots:
    plot_num_subjs(all_datav2, 'Subjects with Post-COVID Data\nEvaluated by Model\n'
                   + ' (Total N=' + str(all_datav2.shape[0]) + ')', struct_var, 'post-covid_allsubj', filepath)

# Reset indices
all_datav2.reset_index(inplace=True, drop=True)

# Calculate age acceleration
agediff_female, agediff_male = calculate_avg_brain_age_acceleration_apply_model(roi_ids, 'allreg', all_datav2,
                                                 struct_var, show_plots, model_dir, spline_order, spline_knots, filepath,
                                                 agemin, agemax, num_permute=0, permute=False, shuffnum=0)

# If calculate bootstrap, run analysis repeatedly for each bootstrap and calculate confidence intervals
if calc_CI_age_acc_bootstrap:

    mean_agediff_boot_f, mean_agediff_boot_m = calculate_avg_brain_age_acceleration_apply_model_bootstrap(roi_ids,
                                                    all_datav2, struct_var, spline_order, spline_knots,
                                                    filepath, agemin, agemax, nbootstrap)

    ageacc_from_bootstraps = {}
    ageacc_from_bootstraps['female'] = mean_agediff_boot_f
    ageacc_from_bootstraps['male'] = mean_agediff_boot_m

    # Write age acceleration from bootstrapping to file
    with open(f"{filepath}/ageacceleration_dictionary {nbootstrap} bootstraps.txt", 'w') as f:
        for key, value in ageacc_from_bootstraps.items():
            f.write('%s:%s\n' % (key, value))

# Show plot of age acceleration for each gender
plot_age_acceleration(filepath, nbootstrap, agediff_female, agediff_male)

plt.show()
mystop=1