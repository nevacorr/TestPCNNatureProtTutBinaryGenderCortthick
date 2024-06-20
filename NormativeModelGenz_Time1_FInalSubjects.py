#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent cortica1 thickness data collected at two time points (before and after the COVID lockdowns).
# This program creates models of cortical thickness change between 9 and 17 years of age for our pre-COVID data and
# stores these models to be applied in another script (Apply_Normative_Model_to_Genz_Time2_Final_Subjects.py)
# to the post-COVID data.
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######

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
from fusiform_spline_plots import plot_data_with_spline_rh_fusiform
from Utility_Functions import write_ages_to_file
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender

struct_var = 'cortthick'
show_plots = 0  #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1 # order of spline to use for model
spline_knots = 2 # number of knots in spline to use in model
perform_train_test_split_precovid = 0  # flag indicating whether to split training set (pre-covid data) into train and
                                       # validations (test) sets. If this is set to 0, the entire training set is used
                                       # for the model and there is no validation set. Regardless of the value of this
                                       # flag, no post-covid data is used in creating or evaluating the normative model.

working_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'

#turn off interactive mode, don't show plots unless plt.show() is specified
plt.ioff()

#load visit 1 (pre-COVID) data
visit=1
brain_good, all_data, roi_ids = load_genz_data(struct_var, visit, working_dir)

#make directories to store files
makenewdir('data/')
makenewdir('data/{}'.format(struct_var))
makenewdir('data/{}/plots'.format(struct_var))
makenewdir('data/{}/ROI_models'.format(struct_var))
makenewdir('data/{}/covariate_files'.format(struct_var))
makenewdir('data/{}/response_files'.format(struct_var))

#remove subject 525 who has an incidental finding
brain_good = brain_good[~brain_good['participant_id'].isin([525])]
all_data = all_data[~all_data['participant_id'].isin([525])]

#replace gender codes 1=male 2=female with binary values (make male=1 and female=0)
all_data.loc[all_data['sex']==2, 'sex'] = 0

#show bar plots with number of subjects per age group in pre-COVID data
if show_nsubject_plots:
    plot_num_subjs(all_data, 'Subjects by Age with Pre-COVID Data\n'
                   '(Total N=' + str(all_data.shape[0]) + ')', struct_var, 'pre-covid_allsubj', working_dir)

########
#Before any modeling, determine which subjects will be included in the training (pre-COVID) and test (post-COVID) analysis.
#Save subject numbers to file.
#Do this only once and then comment the four following lines of code out.
########
#save_test_set_to_file_no_long(struct_var, 9)
#save_test_set_to_file_no_long(struct_var, 11)
#save_test_set_to_file_no_long(struct_var, 13)
#The resulting saved file is named visit1_subjects_excluded_from_normative_model_test_set_{struct_var}_9_11_13.txt

# read in file of subjects in test set at ages 9, 11 and 13
fname='{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(working_dir, struct_var)
subjects_test = pd.read_csv(fname, header=None)

# exclude subjects from the training set who are in test set
brain_good = brain_good[~brain_good['participant_id'].isin(subjects_test[0])]
all_data = all_data[~all_data['participant_id'].isin(subjects_test[0])]

#write subject numbers for training set to file
subjects_training = all_data['participant_id'].tolist()
fname = '{}/visit1_subjects_used_to_create_normative_model_train_set_{}.txt'.format(working_dir, struct_var)
file1 = open(fname, "w")
for subj in subjects_training:
    file1.write(str(subj) + "\n")
file1.close()

# plot number of subjects of each gender by age who are included in training data set
if show_nsubject_plots:
    plot_num_subjs(all_data, 'Subjects by Age with Pre-COVID Data\nUsed to Create Model\n'
                   '(Total N=' + str(all_data.shape[0]) + ')', struct_var, 'pre-covid_norm_model', working_dir)

#drop rows with any missing values
all_data = all_data.dropna()
all_data.reset_index(inplace=True, drop=True)

# separate the brain features (response variables) and predictors (age, gender) in to separate dataframes
all_data_features = all_data.loc[:,roi_ids]
all_data_covariates = all_data[['age', 'agedays', 'sex']]

# If perform_train_test_split_precovid ==1 , split the training set into training and validation set.
# If it is zero, create model based on entire training set
if perform_train_test_split_precovid:
#Split training set into training and validation sets. Training set will be used to create models. Performance will be
# evaluated on the validation set. When performing train-test split, stratify by age and gender
    X_train, X_test, y_train, y_test=train_test_split(all_data_covariates, all_data_features,
                                                      stratify=all_data[['age', 'sex']], test_size=0.2, random_state=42)
else:
#use entire training set to create models
    X_train = all_data_covariates.copy()
    X_test = all_data_covariates.copy()
    y_train = all_data_features.copy()
    y_test = all_data_features.copy()

#identify age range in pre-COVID data to be used for modeling
agemin=X_train['agedays'].min()
agemax=X_train['agedays'].max()

write_ages_to_file(agemin, agemax, struct_var)

# save the subject numbers for the training and validation sets to file
s_index_train = X_train.index.values
s_index_test = X_test.index.values
subjects_train = all_data.loc[s_index_train, 'participant_id'].values
subjects_test = all_data.loc[s_index_test, 'participant_id'].values

# drop the age column from the train and validation data sets because we want to use agedays as a predictor
X_train.drop(columns=['age'], inplace=True)
X_test.drop(columns=['age'], inplace=True)

# change the indices in the train and validation data sets because nan values were dropped above
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

##########
# Set up output directories. Save each brain region to its own text file, organized in separate directories,
# because fpr each response variable Y (brain region) we fit a separate normative mode
##########
for c in y_train.columns:
    y_train[c].to_csv('resp_tr_'+c+'.txt', header=False, index=False)
    X_train.to_csv('cov_tr.txt', sep='\t', header=False, index=False)
    y_train.to_csv('resp_tr.txt', sep='\t', header=False, index=False)
for c in y_test.columns:
    y_test[c].to_csv('resp_te_'+c+'.txt', header=False, index=False)
    X_test.to_csv('cov_te.txt', sep='\t', header=False, index=False)
    y_test.to_csv('resp_te.txt', sep='\t', header=False, index=False)
    if c == 'cortthick-rh-fusiform':
        X_test.to_csv('cov_te_fusiform_validation.txt', sep='\t', header=False, index=False)
        y_test.to_csv('resp_te_fusiform_validation.txt', sep='\t', header=False, index=False)

for i in roi_ids:
    roidirname = 'data/{}/ROI_models/{}'.format(struct_var, i)
    makenewdir(roidirname)
    if i == 'cortthick-rh-fusiform' and perform_train_test_split_precovid == 1:
        shutil.copyfile('cov_te.txt', roidirname + '/cov_te_fusiform_validation.txt')
        shutil.copyfile(f'resp_te_{i}.txt', roidirname + '/resp_te_fusiform_validation.txt')
    resp_tr_filename = "resp_tr_{}.txt".format(i)
    resp_tr_filepath = roidirname + '/resp_tr.txt'
    shutil.copyfile(resp_tr_filename, resp_tr_filepath)
    resp_te_filename = "resp_te_{}.txt".format(i)
    resp_te_filepath = roidirname + '/resp_te.txt'
    shutil.copyfile(resp_te_filename, resp_te_filepath)
    cov_tr_filepath = roidirname + '/cov_tr.txt'
    shutil.copyfile("cov_tr.txt", cov_tr_filepath)
    cov_te_filepath = roidirname + '/cov_te.txt'
    shutil.copyfile("cov_te.txt", cov_te_filepath)

movefiles("resp_*.txt", "data/{}/response_files/".format(struct_var))
movefiles("cov_t*.txt", "data/{}/covariate_files/".format(struct_var))

#  this path is where ROI_models folders are located
data_dir='{}/data/{}/ROI_models/'.format(working_dir, struct_var)

# Create Design Matrix and add in spline basis and intercept for validation and training data
create_design_matrix('test', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)
create_design_matrix('train', agemin, agemax, spline_order, spline_knots, roi_ids, data_dir)

# Create pandas dataframes with header names to save evaluation metrics
blr_metrics=pd.DataFrame(columns=['ROI', 'MSLL', 'EV', 'SMSE','RMSE', 'Rho'])
blr_site_metrics=pd.DataFrame(columns=['ROI', 'y_mean','y_var', 'yhat_mean','yhat_var', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])

# create dataframe with subject numbers to put the Z scores in. Here 'test' refers to the validation set
subjects_test = subjects_test.reshape(-1, 1)
subjects_train = subjects_train.reshape(-1,1)
Z_score_test_matrix = pd.DataFrame(subjects_test, columns=['subject_id_test'])
Z_score_train_matrix = pd.DataFrame(subjects_train, columns=['subject_id_train'])

# Create dataframe to store variances
variance_time1 = pd.DataFrame(subjects_test, columns=['participant_id'])

# Estimate the normative model using a for loop to iterate over brain regions. The estimate function uses a few
# specific arguments that are worth commenting on:
# ●alg=‘blr’: specifies we should use BLR. See Table1 for other available algorithms
# ●optimizer=‘powell’:usePowell’s derivative-free optimization method(faster in this case than L-BFGS)
# ●savemodel=True: do not write out the final estimated model to disk
# ●saveoutput=False: return the outputs directly rather than writing them to disk
# ●standardize=False: do not standardize the covariates or response variable

# Loop through ROIs

for roi in roi_ids:
    print('Running ROI:', roi)
    roi_dir=os.path.join(data_dir, roi)
    model_dir = os.path.join(data_dir, roi, 'Models')
    os.chdir(roi_dir)

    # configure the covariates to use. Change *_bspline_* to *_int_*
    cov_file_tr=os.path.join(roi_dir, 'cov_bspline_tr.txt')
    cov_file_te=os.path.join(roi_dir, 'cov_bspline_te.txt')

    # load train & test response files
    resp_file_tr=os.path.join(roi_dir, 'resp_tr.txt')
    resp_file_te=os.path.join(roi_dir, 'resp_te.txt')

    # calculate a model based on the training data and apply to the validation dataset. If the model is being created
    # from the entire training set, the validation set is simply a copy of the full training set and the purpose of
    # running this function is to creat and save the model, not to evaluate performance. The following are calcualted:
    # the predicted validation set response (yhat_te), the variance of the predicted response (s2_te), the model
    # parameters (nm),the Zscores for the validation data, and other various metrics (metrics_te)
    yhat_te, s2_te, nm, Z_te, metrics_te = estimate(cov_file_tr,resp_file_tr,testresp=resp_file_te,
                                                testcov=cov_file_te,alg='blr',optimizer='powell',
                                                savemodel=True, saveoutput=False,standardize=False)

    variance_time1[roi] = s2_te

    Rho_te=metrics_te['Rho']
    EV_te=metrics_te['EXPV']

    if show_plots:
        #plot y versus y hat for validation data
        plot_y_v_yhat(cov_file_te, resp_file_te, yhat_te, 'Validation Data', struct_var, roi, Rho_te, EV_te)

    #create dummy design matrices for visualizing model
    dummy_cov_file_path_female, dummy_cov_file_path_male = \
        create_dummy_design_matrix(struct_var, agemin, agemax, cov_file_tr, spline_order, spline_knots, working_dir)

    #compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
    plot_data_with_spline('Training Data', struct_var, cov_file_tr, resp_file_tr, dummy_cov_file_path_female,
                          dummy_cov_file_path_male, model_dir, roi, show_plots, working_dir)
    plot_data_with_spline('Validation Data', struct_var, cov_file_te, resp_file_te, dummy_cov_file_path_female,
                          dummy_cov_file_path_male, model_dir, roi, show_plots, working_dir)

    if roi == 'cortthick-rh-fusiform':
        plot_data_with_spline_rh_fusiform('Pre-COVID Subsample ', struct_var, cov_file_tr, resp_file_tr,
                                          dummy_cov_file_path_female, dummy_cov_file_path_male, model_dir, roi,
                                          show_plots, working_dir)
        tmp_cov_file_te = f'{data_dir}/{roi}/cov_te_fusiform_validation.txt'
        tmp_resp_file_te = f'{data_dir}/{roi}/resp_te_fusiform_validation.txt'
        plot_data_with_spline_rh_fusiform('Pre-COVID Subsample Not Used For Model ', struct_var, tmp_cov_file_te, tmp_resp_file_te,
                                      dummy_cov_file_path_female, dummy_cov_file_path_male, model_dir, roi,
                                      show_plots, working_dir)

    #add a row to the blr_metrics dataframe containing ROI, MSLL, EXPV, SMSE, RMSE, and Rho metrics
    blr_metrics.loc[len(blr_metrics)]=[roi, metrics_te['MSLL'][0],
            metrics_te['EXPV'][0], metrics_te['SMSE'][0], metrics_te['RMSE'][0],metrics_te['Rho'][0]]

    # load test (pre-COVID validation) data
    X_te=np.loadtxt(cov_file_te)
    y_te=np.loadtxt(resp_file_te)
    y_te=y_te[:, np.newaxis] # make sure it is a 2-d array

    y_mean_te=np.mean(y_te)
    y_var_te=np.var(y_te)
    yhat_mean_te=np.mean(yhat_te)
    yhat_var_te=np.var(yhat_te)

    metrics_te=evaluate(y_te, yhat_te, s2_te,y_mean_te, y_var_te)

    blr_site_metrics.loc[len(blr_site_metrics)]=[roi, y_mean_te,y_var_te,yhat_mean_te,yhat_var_te,metrics_te['MSLL'][0],
                                                 metrics_te['EXPV'][0],metrics_te['SMSE'][0], metrics_te['RMSE'][0],
                                                 metrics_te['Rho'][0]]
    #store z score for ROI validation set
    Z_score_test_matrix[roi] = Z_te

blr_site_metrics.to_csv('{}/data/{}/blr_metrics_{}.txt'.format(working_dir, struct_var, struct_var), index=False)

#save validation z scores to file
Z_score_test_matrix.to_csv('{}/data/{}/Z_scores_by_region_validation_set.txt'. format(working_dir, struct_var),
                           index=False)

# write variance to file
variance_time1.to_csv(f'{working_dir}/variance in predictions for pre-covid (time1) data', index=False)

# if perform_train_test_split_precovid:
#     Z_score_test = Z_score_test_matrix.copy()
#     Z_score_test.rename(columns = {'subject_id_test': 'participant_id'}, inplace=True)
#     plot_and_compute_zcores_by_gender(Z_score_test, struct_var, roi_ids)

##########
# Display plots of Rho and EV for validation set
##########

blr_metrics.sort_values(by=['Rho'], inplace=True, ignore_index=True)
barplot_performance_values(struct_var, 'Rho', blr_metrics, spline_order, spline_knots, 'Validation Set', working_dir)
blr_metrics.sort_values(by=['EV'], inplace=True, ignore_index=True)
barplot_performance_values(struct_var, 'EV', blr_metrics, spline_order, spline_knots, 'Validation Set', working_dir)
plt.show()

mystop=1