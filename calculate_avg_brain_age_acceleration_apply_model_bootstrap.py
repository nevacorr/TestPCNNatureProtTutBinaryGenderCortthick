###
# Uses bootstrapping to calculate confidence intervals for age acceleration
##@


import pandas as pd
import matplotlib.pyplot as plt
from plot_num_subjs import plot_num_subjs
from pcntoolkit.normative import estimate, evaluate
from plot_num_subjs import plot_num_subjs
from Utility_Functions import create_design_matrix, plot_data_with_spline
from Utility_Functions import create_dummy_design_matrix
from Utility_Functions import plot_y_v_yhat, makenewdir, movefiles
from Utility_Functions import write_ages_to_file, read_ages_from_file
from Load_Genz_Data import load_genz_data
import shutil
import os
from normative_edited import predict
from calculate_brain_age_acceleration import calculate_age_acceleration

def calculate_avg_brain_age_acceleration_apply_model_bootstrap(roi_ids, all_data, struct_var, spline_order, spline_knots,
                                                               working_dir, agemin, agemax, n_bootstraps):

    #############################  Apply Normative Model to Post-COVID Data ####################

    mean_agediff_boot_f= []
    mean_agediff_boot_m = []

    all_data_orig_samp = all_data

    for b in range(n_bootstraps):

        print(f'bootstrap {b} of {n_bootstraps}')

        # remove output bootstrap directories if they already exist from previous iterations
        dirpath = f'{working_dir}/predict_files_bootstrap'
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)

        # make file directories for output
        makenewdir('{}/predict_files_bootstrap/'.format(working_dir))
        makenewdir('{}/predict_files_bootstrap/avgct_{}'.format(working_dir, struct_var))
        makenewdir('{}/predict_files_bootstrap/avgct_{}/plots'.format(working_dir, struct_var))
        makenewdir('{}/predict_files_bootstrap/avgct_{}/ROI_models'.format(working_dir, struct_var))
        makenewdir('{}/predict_files_bootstrap/avgct_{}/covariate_files'.format(working_dir, struct_var))
        makenewdir('{}/predict_files_bootstrap/avgct_{}/response_files'.format(working_dir, struct_var))

         # resample the dataframe with replacement, stratify by age
        all_data = all_data_orig_samp.groupby('age', group_keys=False).apply(lambda x: x.sample(frac=1, replace=True, axis=0))

        all_data.reset_index(inplace=True, drop=True)

        #specify which columns of dataframe to use as covariates
        X_test = all_data[['agedays', 'sex']]

        #make a matrix of response variables, one for each brain region
        y_test = all_data.loc[:, roi_ids]

        #average cortical thickness across all regions for each subject
        y_test = y_test.mean(axis=1).to_frame()
        y_test.rename(columns={0:'avgcortthick'},  inplace=True)

        #specify paths
        training_dir = '{}/avgct_allreg/{}/ROI_models/'.format(working_dir, struct_var)
        out_dir = '{}/predict_files_bootstrap/avgct_{}/ROI_models/'.format(working_dir, struct_var)
        #  this path is where ROI_models folders are located
        predict_files_dir = '{}/predict_files_bootstrap/avgct_{}/ROI_models/'.format(working_dir, struct_var)

        ##########
        # Create output directories for each region and place covariate and response files for that region in  each directory
        ##########
        for c in y_test.columns:
            y_test[c].to_csv(f'{working_dir}/resp_te_'+c+'.txt', header=False, index=False)
            X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
            y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

        for i in ['avgcortthick']:
            roidirname = '{}/predict_files_bootstrap/avgct_{}/ROI_models/{}'.format(working_dir, struct_var, i)
            makenewdir(roidirname)
            resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
            resp_te_filepath = roidirname + '/resp_te.txt'
            shutil.copyfile(resp_te_filename, resp_te_filepath)
            cov_te_filepath = roidirname + '/cov_te.txt'
            shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

        movefiles("{}/resp_*.txt".format(working_dir), "{}/predict_files_bootstrap/avgct_{}/response_files/"
                  .format(working_dir, struct_var))
        movefiles("{}/cov_t*.txt".format(working_dir), "{}/predict_files_bootstrap/avgct_{}/covariate_files/"
                  .format(working_dir, struct_var))

        # Create dataframe to store Zscores
        Z_time2 = pd.DataFrame()
        Z_time2['participant_id'] = all_data['participant_id'].copy()
        Z_time2.reset_index(inplace=True, drop = True)

        ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

        # Create design matrices for all regions and save files in respective directories
        create_design_matrix('test', agemin, agemax, spline_order, spline_knots, ['avgcortthick'], predict_files_dir)

        for roi in ['avgcortthick']:
            print('Running ROI:', roi)
            roi_dir=os.path.join(predict_files_dir, roi)
            model_dir = os.path.join(training_dir, roi, 'Models')
            os.chdir(roi_dir)

            # Configure the covariates to use.
            cov_file_te=os.path.join(roi_dir, 'cov_bspline_te.txt')

            # Load test response files
            resp_file_te=os.path.join(roi_dir, 'resp_te.txt')

            # Make predictions
            yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

            Z_time2[roi] = Z

            # Create dummy design matrices
            dummy_cov_file_path_female, dummy_cov_file_path_male = \
                create_dummy_design_matrix(struct_var, agemin, agemax, cov_file_te, spline_order,
                                                      spline_knots, working_dir)

            # Calculate brain age acceleration
            mean_f, mean_m = (calculate_age_acceleration(struct_var, roi_dir, yhat_te, model_dir, roi,
                                                                dummy_cov_file_path_female, dummy_cov_file_path_male,
                                                                plotgap=0))

            mean_agediff_boot_f.append(mean_f)
            mean_agediff_boot_m.append(mean_m)

        mystop=1

    return mean_agediff_boot_f, mean_agediff_boot_m
