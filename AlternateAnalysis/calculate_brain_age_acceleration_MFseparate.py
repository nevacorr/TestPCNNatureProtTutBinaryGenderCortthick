#####
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######

import numpy as np
import os
from Utility_Functions_MF_Separate import fit_regression_model_dummy_data

days_to_years_factor=365.25

def calculate_age_acceleration_one_gender(roi_dir, model_dir, dummy_cov_file_path):

    # Load age and gender (predictors)
    actual_age = np.loadtxt(os.path.join(roi_dir, 'cov_te.txt'))

    # Load measured struct_var
    actual_struct = np.loadtxt(os.path.join(roi_dir, 'resp_te.txt'))

    # Calculate coefficients for linear model equation
    slope, intercept= fit_regression_model_dummy_data(model_dir, dummy_cov_file_path)

    # Por every subject, calculate predicted age based on model equation
    predicted_age = (actual_struct - intercept)/slope

    # Subtract mean average age from mean predicted age for each age group
    mean_agediff = np.mean(np.subtract(predicted_age, actual_age))/days_to_years_factor

    return mean_agediff


