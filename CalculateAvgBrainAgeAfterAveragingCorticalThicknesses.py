#####
# This program calculates brain age acceleration based on the adolescent data. It averages cortical thickness across
# all brain regions. It fits a model on the precovid data and evaluates the model on the post-covid data.
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######

import pandas as pd
import matplotlib.pyplot as plt
from Load_Genz_Data import load_genz_data
from plot_num_subjs import plot_num_subjs
from calculate_avg_brain_age_acc_across_select_regions import calculate_avg_brain_age_acceleration_across_select_regions

struct_var = 'cortthick'
show_plots = 1  #set to 1 to show training and test data y vs yhat and spline fit plots. Set to 0 to save to file.
show_nsubject_plots = 1 #set to 1 to show number of subjects in analysis
spline_order = 1
spline_knots = 2
filepath = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'

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
f_sigreg=[]
m_sigreg=[]


# returns the age acceleration for the males and females when cortthick is averaged across the entire brain
avg_all_regions_age_diff_f, avg_all_regions_age_diff_m = calculate_avg_brain_age_acceleration_across_select_regions('allreg',
                        f_sigreg, m_sigreg, all_data, all_data_covariates, all_data_features, struct_var, show_plots,
                        show_nsubject_plots, spline_order, spline_knots, filepath)

plt.show()
mystop=1