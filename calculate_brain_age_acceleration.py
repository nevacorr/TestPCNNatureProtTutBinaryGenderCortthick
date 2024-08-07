#####
# Author: Neva M. Corrigan
# Returns age acceleration for males and females for post-covid data based on pre-covid model.
# Date: 21 February, 2024
######

import numpy as np
import pandas as pd
from scipy.stats import sem
import os
from Utility_Functions import fit_regression_model_dummy_data
from matplotlib import pyplot as plt

days_to_years_factor=365.25

def calculate_age_acceleration(struct_var, roi_dir, yhat, model_dir, roi,
                               dummy_cov_file_path_female, dummy_cov_file_path_male, plotgap):

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

    if plotgap:
        age_gap_f =(predicted_age_f - actual_age_f)/days_to_years_factor
        age_gap_m =(predicted_age_m - actual_age_m)/days_to_years_factor

        age_gap_df_f = pd.DataFrame()
        age_gap_df_m = pd.DataFrame()
        age_gap_df_f['actual_age']=actual_age_f/days_to_years_factor
        age_gap_df_m['actual_age']=actual_age_m/days_to_years_factor
        age_gap_df_f['age_gap'] = age_gap_f
        age_gap_df_m['age_gap'] = age_gap_m
        age_gap_df_f['actual_age_int'] = np.floor(age_gap_df_f['actual_age'].astype('int'))
        age_gap_df_m['actual_age_int'] = np.floor(age_gap_df_m['actual_age'].astype('int'))

        mean_vals_f = age_gap_df_f.groupby('actual_age_int')[['actual_age', 'age_gap']].mean().reset_index()
        mean_vals_m = age_gap_df_m.groupby('actual_age_int')[['actual_age', 'age_gap']].mean().reset_index()
        sem_vals_f = age_gap_df_f.groupby('actual_age_int')[['actual_age', 'age_gap']].sem().reset_index()
        sem_vals_m = age_gap_df_m.groupby('actual_age_int')[['actual_age', 'age_gap']].sem().reset_index()

        # Plot mean actual age with error bars for females
        plt.errorbar(mean_vals_f['actual_age_int'], mean_vals_f['age_gap'], yerr=sem_vals_f['age_gap'], color='g',fmt='o',
                     capsize=5, label='female')

        # Plot mean actual age with error bars for males
        plt.errorbar(mean_vals_m['actual_age_int'] + 0.1, mean_vals_m['age_gap'], yerr=sem_vals_m['age_gap'], color='b', fmt='o',
                     capsize=5, label='male')

        plt.xlabel('Actual Age (years)')
        plt.ylabel('Age Gap (years)')
        plt.title('Mean Age Gap by Age Group and Gender')
        plt.legend()
        plt.show()

    #subtract mean average age from mean predicted age for each age group
    mean_agediff_f = np.mean(np.subtract(predicted_age_f, actual_age_f))/days_to_years_factor
    mean_agediff_m = np.mean(np.subtract(predicted_age_m, actual_age_m))/days_to_years_factor

    return mean_agediff_f, mean_agediff_m


