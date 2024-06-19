#####
# This program calculates brain age acceleration based on the adolescent data. It averages cortical thickness across
# all brain regions. It fits a model on the precovid data and evaluates the model on the post-covid data.
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######

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
filepath = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
num_permute = 1000    #number of permutations to use in calculating signifiance of sex difference in age acceleration
calc_age_acc_diff_permute = 0
calc_CI_age_acc_bootstrap = 0
nbootstrap = 1000

#turn off interactive mode, don't show plots unless plt.show() is specified
plt.ioff()

visit=1
brain_good, all_data, roi_ids = load_genz_data(struct_var, visit, filepath)

#remove subject 525 who has an incidental finding
all_data = all_data[~all_data['participant_id'].isin([525])]

# read in file of subjects in test set at ages 9, 11 and 13
fname='{}/visit2_all_subjects_used_in_test_set_cortthick.txt'.format(filepath, struct_var)
subjects_test = pd.read_csv(fname, header=None)

# exclude subjects at 9, 11 and 13 who are in test set
all_data = all_data[~all_data['participant_id'].isin(subjects_test[0])]

#replace gender with gender=0 female gender =1 male
all_data.loc[all_data['sex']==2, 'sex'] = 0

# plot number of subjects of each gender by age who are included in training data set
if show_nsubject_plots:
    plot_num_subjs(all_data, 'Subjects by Age with Pre-COVID Data\nUsed to Create Model\n'
                   '(Total N=' + str(all_data.shape[0]) + ')', struct_var, 'pre-covid_norm_model', filepath)

#drop rows with any missing values
all_data = all_data.dropna()
all_data.reset_index(inplace=True, drop=True)

# separate the brain features (response variables) and predictors (age, gender) in to separate dataframes
all_data_features_orig = all_data.loc[:,roi_ids]
all_data_covariates = all_data[['age', 'agedays', 'sex']]


# average cortical thickness across all regions for each subject
all_data_features = all_data_features_orig.mean(axis=1).to_frame()
all_data_features.rename(columns={0:'avgcortthick'},  inplace=True)

# returns the age acceleration for the males and females when cortthick is averaged across the entire brain
model_dir, agemin, agemax = calculate_avg_brain_age_acceleration_make_model('allreg',
                        all_data, all_data_covariates, all_data_features, struct_var, show_plots,
                        show_nsubject_plots, spline_order, spline_knots, filepath)

# specify visit number
visit = 2
# load all brain and behavior data for visit 2
brain_good, all_datav2, roi_ids = load_genz_data(struct_var, visit, filepath)

fname = '{}/visit2_all_subjects_used_in_test_set_{}.txt'.format(filepath, struct_var)
my_file = open(fname, 'r')
test_subjects_txt = my_file.read()
test_subjects = test_subjects_txt.split("\n")
my_file.close()
while ("" in test_subjects):
    test_subjects.remove("")
test_subjects = [int(i) for i in test_subjects]

all_datav2 = all_datav2[all_datav2['participant_id'].isin(test_subjects)]

# replace gender with binary gender
all_datav2.loc[all_datav2['sex'] == 2, 'sex'] = 0

# show number of subjects by gender and age
if show_nsubject_plots:
    plot_num_subjs(all_datav2, 'Subjects with Post-COVID Data\nEvaluated by Model\n'
                   + ' (Total N=' + str(all_datav2.shape[0]) + ')', struct_var, 'post-covid_allsubj', filepath)

# reset indices
all_datav2.reset_index(inplace=True, drop=True)

agediff_female, agediff_male = calculate_avg_brain_age_acceleration_apply_model(roi_ids, 'allreg', all_datav2,
                                                 struct_var, show_plots, model_dir, spline_order, spline_knots, filepath,
                                                 agemin, agemax, num_permute=0, permute=False, shuffnum=0)

mean_agediff_permuted_df = pd.DataFrame()
if calc_age_acc_diff_permute:
    for i_permute in range(num_permute):
        print(f'i_permute = {i_permute}')

        female_agediff, male_agediff = calculate_avg_brain_age_acceleration_apply_model(roi_ids, 'allreg', all_datav2,
                                                 struct_var, 0, model_dir, spline_order, spline_knots, filepath, agemin,
                                                 agemax, num_permute=num_permute, permute=True, shuffnum=i_permute)

        m = pd.DataFrame(columns=['female', 'male'])
        m.loc[0, 'female'] = female_agediff
        m.loc[0, 'male'] = male_agediff
        mean_agediff_permuted_df = pd.concat([mean_agediff_permuted_df, m], ignore_index=True)

    # Determine percentile of mean_agediff
    mean_age_diff_permuted_female = mean_agediff_permuted_df['female'].to_numpy()
    mean_age_diff_permuted_male = mean_agediff_permuted_df['male'].to_numpy()
    sex_age_diff_array = np.squeeze(mean_age_diff_permuted_female - mean_age_diff_permuted_male)

    empirical_gender_diff = agediff_female - agediff_male
    # Find out what percentile value is with respect to array
    percentile = percentileofscore(sex_age_diff_array, empirical_gender_diff)
    # # Print the percentile
    print("The percentile of", empirical_gender_diff, "with respect to the array is:", percentile)
    # Write empirical percentile and permutation results for sex diff array to file
    # append empirical percentile to end of array
    sex_age_diff_array = np.append(sex_age_diff_array, empirical_gender_diff)
    # Save array to text file
    np.savetxt(f'{filepath}/sex acceleration distribution.txt', sex_age_diff_array)

# if calc_CI_age_acc_bootstrap:
#
#     mean_agediff_boot_f, mean_agediff_boot_m = calculate_avg_brain_age_acceleration_apply_model_bootstrap(roi_ids, all_datav2, struct_var,
#                                                        spline_order, spline_knots,
#                                                        filepath, agemin, agemax, nbootstrap)
#
#     ageacc_from_bootstraps = {}
#     ageacc_from_bootstraps['female'] = mean_agediff_boot_f
#     ageacc_from_bootstraps['male'] = mean_agediff_boot_m
#
#     # Write age acceleration from bootstrapping to file
#     with open(f"{filepath}/ageacceleration_dictionary {nbootstrap} bootstraps.txt", 'w') as f:
#         for key, value in ageacc_from_bootstraps.items():
#             f.write('%s:%s\n' % (key, value))

plot_age_acceleration(filepath, nbootstrap, agediff_female, agediff_male)



plt.show()
mystop=1