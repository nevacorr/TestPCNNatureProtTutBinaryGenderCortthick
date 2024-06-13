import pandas as pd
import numpy as np
from Utility_Functions import makenewdir, movefiles, create_dummy_design_matrix, fit_regression_model_dummy_data
from Utility_Functions import plot_data_with_spline, create_design_matrix
import shutil
from normative_edited import predict
import os

# Define your custom functions for calculating cortical thicknesses based on spline model for both genders
def calculate_new_column_male(age, slope_m, intercept_m):
    # Equation for males
    return slope_m*age + intercept_m

def calculate_new_column_female(age, slope_f, intercept_f):
    # Equation for females
    return slope_f*age + intercept_f

def apply_normmodel_postcovid(all_data, roi_ids, working_dir, struct_var, agemin, agemax, spline_order, spline_knots,
                              training_dir, out_dir, predict_files_dir):
    #specify which columns of dataframe to use as covariates
    X_test = all_data[['agedays', 'sex']]

    #make a matrix of response variables, one for each brain region
    y_test = all_data.loc[:, roi_ids]

    ##########
    # Create output directories for each region and place covariate and response files for that region in  each directory
    ##########
    for c in y_test.columns:
        y_test[c].to_csv(f'{working_dir}/resp_te_'+c+'.txt', header=False, index=False)
        X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
        y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

    for i in roi_ids:
        roidirname = '{}/predict_files_bootstrap/{}/ROI_models/{}'.format(working_dir, struct_var, i)
        makenewdir(roidirname)
        resp_te_filename = '{}/resp_te_{}.txt'.format(working_dir, i)
        resp_te_filepath = roidirname + '/resp_te.txt'
        shutil.copyfile(resp_te_filename, resp_te_filepath)
        cov_te_filepath = roidirname + '/cov_te.txt'
        shutil.copyfile(f'{working_dir}/cov_te.txt', cov_te_filepath)

    movefiles(f"{working_dir}/resp_*.txt", "{}/predict_files_bootstrap/{}/response_files/".format(working_dir, struct_var))
    movefiles(f"{working_dir}/cov_t*.txt", "{}/predict_files_bootstrap/{}/covariate_files/".format(working_dir, struct_var))

    # Create Design Matrix and add in spline basis and intercept
    create_design_matrix('test', agemin, agemax, spline_order, spline_knots, roi_ids, out_dir)

    # Create dataframe to store Zscores
    Z_time2 = pd.DataFrame()
    Z_time2['participant_id'] = all_data['participant_id'].copy()
    Z_time2.reset_index(inplace=True, drop = True)

    # Create dataframe to store variances
    variance_time2 = pd.DataFrame()
    variance_time2['participant_id'] = all_data['participant_id'].copy()
    variance_time2.reset_index(inplace=True, drop = True)

    ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

    #create design matrices for all regions and save files in respective directories
    create_design_matrix('test', agemin, agemax, spline_order, spline_knots, roi_ids, predict_files_dir)

    effect_size_by_region = pd.DataFrame(columns=roi_ids)
    for roi in roi_ids:

        roi_dir=os.path.join(predict_files_dir, roi)
        model_dir = os.path.join(training_dir, roi, 'Models')
        os.chdir(roi_dir)

        # configure the covariates to use.
        cov_file_te=os.path.join(roi_dir, 'cov_bspline_te.txt')

        # load test response files
        resp_file_te=os.path.join(roi_dir, 'resp_te.txt')

        # make predictions
        yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

        variance_time2[roi] = s2_te

        Z_time2[roi] = Z

        #create dummy design matrices
        dummy_cov_file_path_female, dummy_cov_file_path_male = \
            create_dummy_design_matrix(struct_var, agemin, agemax, cov_file_te, spline_order, spline_knots, working_dir)

        # plot_data_with_spline('Postcovid (Test) Data ', struct_var, cov_file_te, resp_file_te, dummy_cov_file_path_female,
        #                           dummy_cov_file_path_male, model_dir, roi, show_plots, working_dir)

        slope_f, intercept_f, slope_m, intercept_m = (
                fit_regression_model_dummy_data(model_dir, dummy_cov_file_path_female, dummy_cov_file_path_male))

        cov_data_roi = pd.read_csv(cov_file_te, header=None, sep=' ',names=['AgeDays', 'gender', 'Intercept', 'Spline1', 'Spline2'])
        resp_data_roi = pd.read_csv(resp_file_te, header=None, sep=' ', names=[roi])

        age_data_f = cov_data_roi.loc[cov_data_roi['gender']==0, 'AgeDays']
        age_data_m = cov_data_roi.loc[cov_data_roi['gender']==1, 'AgeDays']

        #calculate cortical thickness values as calculated for the model
        y_from_model = pd.DataFrame()
        y_from_model['ct'] = cov_data_roi.apply(
            lambda row: calculate_new_column_male(row['AgeDays'], slope_m, intercept_m) if row[1] == 0
                                else calculate_new_column_female(row['AgeDays'], slope_f, intercept_f), axis=1)
        effect_df = pd.DataFrame()
        effect_df['AgeDays'] = cov_data_roi['AgeDays'].copy()
        effect_df['AgeGrp'] = effect_df['AgeDays']/365.25
        effect_df['AgeGrp'] = effect_df['AgeGrp'].astype(int)
        effect_df['gender'] = cov_data_roi['gender'].copy()
        effect_df['measured_CT'] = resp_data_roi[roi].copy()
        effect_df['CT_from_model'] = y_from_model['ct'].copy()
        effect_df['predictive_mean'] = yhat_te.copy()
        effect_df['predictive_variance'] = s2_te.copy()

        mean_measured_CT = effect_df.groupby('gender')['measured_CT'].mean()
        mean_CT_from_model = effect_df.groupby('gender')['CT_from_model'].mean()
        mean_variance = effect_df.groupby('gender')['predictive_variance'].mean()
        mean_stdev = np.sqrt(mean_variance)

        effect_size = Z*sqrt(s2_te)+yhat_te
        effect_size_by_region.loc[:,roi] = effect_size

    return effect_size_by_region
