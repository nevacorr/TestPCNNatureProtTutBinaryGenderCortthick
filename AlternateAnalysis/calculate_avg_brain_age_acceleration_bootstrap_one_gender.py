###
# Uses bootstrapping to calculate confidence intervals for age acceleration
##@


import pandas as pd
import matplotlib.pyplot as plt
from plot_num_subjs import plot_num_subjs
from pcntoolkit.normative import estimate, evaluate
from plot_num_subjs import plot_num_subjs
from Utility_Functions_MF_Separate import create_design_matrix_one_gender, plot_data_with_spline_one_gender
from Utility_Functions_MF_Separate import create_dummy_design_matrix_one_gender
from Utility_Functions_MF_Separate import plot_y_v_yhat_one_gender, makenewdir, movefiles
from Utility_Functions_MF_Separate import write_ages_to_file_by_gender, read_ages_from_file
from Load_Genz_Data import load_genz_data
import shutil
import os
from normative_edited import predict
from calculate_brain_age_acceleration_MFseparate import calculate_age_acceleration_one_gender

def calculate_avg_brain_age_acceleration_one_gender_apply_model_bootstrap(gender, orig_struct_var, show_nsubject_plots,
                                                               show_plots, spline_order, spline_knots,
                                                               orig_data_dir, working_dir, n_bootstraps):

    #############################  Apply Normative Model to Post-COVID Data ####################

    # Load all brain and behavior data for visit 2
    visit = 2
    brain_good, all_data, roi_ids = load_genz_data(orig_struct_var, visit, orig_data_dir)

    # Load brain and behavior data for visit 1
    visit = 1
    brain_v1, all_v1, roi_v1 = load_genz_data(orig_struct_var, visit, orig_data_dir)

    # Extract subject numbers from visit 1 and find subjects in visit 2 that aren't in visit 1
    # subjects_visit1 = all_v1['participant_id']
    rows_in_v2_but_not_v1 = all_data[~all_data['participant_id'].isin(all_v1['participant_id'])].dropna()
    subjs_in_v2_not_v1 = rows_in_v2_but_not_v1['participant_id'].copy()
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1.astype(int)

    # Only keep subjects at 12, 14 and 16 years of age (subject numbers <400) because cannot model does not cover
    # older ages
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1[subjs_in_v2_not_v1 < 400]

    if gender == 'male':
        # Keep only data for males
        all_data = all_data.loc[all_data['sex'] == 1]
        struct_var = 'cortthick_male'
    else:
        # Keep only data for females
        all_data = all_data.loc[all_data['sex'] == 2]
        struct_var = 'cortthick_female'

    # Remove sex column
    all_data = all_data.drop(columns=['sex'])

    # Only include subjects that were not in the training set
    fname='{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(orig_data_dir, orig_struct_var)
    subjects_to_include = pd.read_csv(fname, header=None)
    subjects_to_include = pd.concat([subjects_to_include, subjs_in_v2_not_v1])
    # brain_good = brain_good[brain_good['participant_id'].isin(subjects_to_include[0])]
    all_data = all_data[all_data['participant_id'].isin(subjects_to_include[0])]

    # Reset indices
    all_data.reset_index(inplace=True, drop=True)

    # Read agemin and agemax from file
    agemin, agemax = read_ages_from_file(working_dir, struct_var)

    # Show number of subjects by gender and age
    if gender == "female":
        genstring = 'Female'
    elif gender == "male":
        genstring = 'Male'
    if show_nsubject_plots:
        plot_num_subjs(all_data, gender, f'{genstring} Subjects with Post-COVID Data\nEvaluated by Model\n'
                       + ' (Total N=' + str(all_data.shape[0]) + ')', struct_var, 'post-covid_allsubj', working_dir)

    mean_agediff_boot = []

    all_data_orig_samp = all_data

    for b in range(n_bootstraps):

        print(f'bootstrap {b} of {n_bootstraps}')

        # remove output bootstrap directories if they already exist from previous iterations
        dirpath = f'{working_dir}/predict_files_bootstrap'
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)

        # make file diretories for output
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
        X_test = all_data[['agedays']]

        #make a matrix of response variables, one for each brain region
        y_test = all_data.loc[:, roi_ids]

        #average cortical thickness across all regions for each subject
        y_test = y_test.mean(axis=1).to_frame()
        y_test.rename(columns={0:'avgcortthick'},  inplace=True)

        #specify paths
        training_dir = '{}/data/avgct_{}/ROI_models/'.format(working_dir, struct_var)
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
        create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, ['avgcortthick'], predict_files_dir)

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
            dummy_cov_file_path = \
                create_dummy_design_matrix_one_gender(struct_var, agemin, agemax, cov_file_te, spline_order,
                                                      spline_knots, working_dir)

            plot_data_with_spline_one_gender(gender, 'Postcovid (Test) Data ', struct_var, cov_file_te, resp_file_te, dummy_cov_file_path,
                                      model_dir, roi, show_plots, working_dir)

            # Calculate brain age acceleration
            mean_agediff_boot.append(calculate_age_acceleration_one_gender(roi_dir, model_dir, dummy_cov_file_path))

        if show_plots:
            plt.show()

        mystop=1

    return mean_agediff_boot


