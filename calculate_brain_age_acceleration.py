#####
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######

import numpy as np
import os
from Utility_Functions import fit_regression_model_dummy_data

days_to_years_factor=365.25

def calculate_age_acceleration(struct_var, roi_dir, yhat, model_dir, roi,
                               dummy_cov_file_path_female, dummy_cov_file_path_male):

    #load age and gender (predictors)
    predictors = np.loadtxt(os.path.join(roi_dir, 'cov_te.txt'))
    #load measured struct_var
    actual_struct = np.loadtxt(os.path.join(roi_dir, 'resp_te.txt'))
    predicted_struct = yhat

    #separate age and gender into separate variables
    actual_age = predictors[:,0]
    gender = predictors[:,1]

    #find indexes of male and female subjects
    female_ind = np.where(gender==0)
    male_ind = np.where(gender==1)

    #create arrays of actual age and actual structvar for males and females
    actual_age_f = actual_age[female_ind].copy()
    actual_age_m = actual_age[male_ind].copy()
    actual_struct_f = actual_struct[female_ind].copy()
    actual_struct_m = actual_struct[male_ind].copy()

    slope_f, intercept_f, slope_m, intercept_m = fit_regression_model_dummy_data(model_dir,
                                                                dummy_cov_file_path_female, dummy_cov_file_path_male)

    #for every female subject, calculate predicted age
    predicted_age_f = (actual_struct_f - intercept_f)/slope_f
    predicted_age_m = (actual_struct_m - intercept_m)/slope_m

    avg_actual_str_f = np.mean(actual_struct_f)
    avg_actual_str_m = np.mean(actual_struct_m)

    avg_predicted_age_f = np.mean(predicted_age_f)/days_to_years_factor
    avg_predicted_age_m = np.mean(predicted_age_m)/days_to_years_factor

    avg_actual_age_f = np.mean(actual_age_f)/days_to_years_factor
    avg_actual_age_m = np.mean(actual_age_m)/days_to_years_factor

    #subtract mean average age from mean predicted age for each age group
    mean_agediff_f = np.mean(np.subtract(predicted_age_f, actual_age_f))/days_to_years_factor
    mean_agediff_m = np.mean(np.subtract(predicted_age_m, actual_age_m))/days_to_years_factor

    return mean_agediff_f, mean_agediff_m


