import pandas as pd
import os
import shutil
from pcntoolkit.normative import estimate, evaluate, predict
from Utility_Functions import create_design_matrix, plot_data_with_spline, create_dummy_design_matrix
from Utility_Functions import plot_y_v_yhat, makenewdir, movefiles, write_list_of_lists, read_list_of_lists
from normative_edited import predict
from calculate_brain_age_acceleration import calculate_age_acceleration
from plot_and_compute_zdistributions import plot_and_compute_zcores_by_gender
import matplotlib.pyplot as plt
import random
from plot_num_subjs import plot_num_subjs
from Load_Genz_Data import load_genz_data

def calculate_avg_brain_age_acceleration_make_model(desc_string, all_data, all_data_covariates, all_data_features,
                                                               struct_var, show_plots, show_nsubject_plots,
                                                               spline_order, spline_knots, filepath):

    makenewdir('{}/avgct_{}/'.format(filepath, desc_string))
    makenewdir('{}/avgct_{}/{}'.format(filepath, desc_string, struct_var))
    makenewdir('{}/avgct_{}/{}/plots'.format(filepath, desc_string, struct_var))
    makenewdir('{}/avgct_{}/{}/ROI_models'.format(filepath, desc_string, struct_var))
    makenewdir('{}/avgct_{}/{}/covariate_files'.format(filepath, desc_string, struct_var))
    makenewdir('{}/avgct_{}/{}/response_files'.format(filepath, desc_string, struct_var))

    X_train = all_data_covariates.copy()
    y_train = all_data_features.copy()

    agemin=X_train['agedays'].min()
    agemax=X_train['agedays'].max()

    # save the subject numbers for the training set
    s_index_train = X_train.index.values
    subjects_train = all_data.loc[s_index_train, 'participant_id'].values

    # drop the age column because we want to use agedays as a predictor
    X_train.drop(columns=['age'], inplace=True)

    # reset the indices in the train data set
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    ##########
    # Set up output directories. Save each brain region to its own text file, organized in separate directories,
    # because fpr each response variable Y (brain region) we fit a separate normative mode
    ##########
    for c in y_train.columns:
        y_train[c].to_csv(f'{filepath}/resp_tr_'+c+'.txt', header=False, index=False)
        X_train.to_csv(f'{filepath}/cov_tr.txt', sep='\t', header=False, index=False)
        y_train.to_csv(f'{filepath}/resp_tr.txt', sep='\t', header=False, index=False)

    for i in ['avgcortthick']:
        roidirname = '{}/avgct_{}/{}/ROI_models/{}'.format(filepath, desc_string, struct_var, i)
        makenewdir(roidirname)
        resp_tr_filename = "{}/resp_tr_{}.txt".format(filepath, i)
        resp_tr_filepath = roidirname + '/resp_tr.txt'
        shutil.copyfile(resp_tr_filename, resp_tr_filepath)
        cov_tr_filepath = roidirname + '/cov_tr.txt'
        shutil.copyfile(f"{filepath}/cov_tr.txt", cov_tr_filepath)

    movefiles("{}/resp_*.txt", "{}/avgct_{}/{}/response_files/".format(filepath, filepath, desc_string, struct_var))
    movefiles("{}/cov_t*.txt", "{}/avgct_{}/{}/covariate_files/".format(filepath, filepath, desc_string, struct_var))

    #  this path is where ROI_models folders are located
    data_dir='{}/avgct_{}/{}/ROI_models/'.format(filepath, desc_string, struct_var)

    # Create Design Matrix and add in spline basis and intercept
    create_design_matrix('train', agemin, agemax, spline_order, spline_knots, ['avgcortthick'], data_dir)

    # Create pandas dataframes with header names to save evaluation metrics
    blr_metrics=pd.DataFrame(columns=['ROI', 'MSLL', 'EV', 'SMSE','RMSE', 'Rho'])
    blr_site_metrics=pd.DataFrame(columns=['ROI', 'y_mean','y_var', 'yhat_mean','yhat_var', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'])

    # create dataframe with subject numbers to put the Z scores  in
    subjects_train = subjects_train.reshape(-1,1)
    Z_score_train_matrix = pd.DataFrame(subjects_train, columns=['subject_id_train'])

    # Estimate the normative model using a for loop to iterate over brain regions. The estimate function uses a few specific arguments that are worth commenting on:
    # ●alg=‘blr’: specifies we should use BLR. See Table1 for other available algorithms
    # ●optimizer=‘powell’:usePowell’s derivative-free optimization method(faster in this case than L-BFGS)
    # ●savemodel=True: do not write out the final estimated model to disk
    # ●saveoutput=False: return the outputs directly rather than writing them to disk
    # ●standardize=False: do not standardize the covariates or response variable

    # Loop through ROIs

    for roi in ['avgcortthick']:
        print('Running ROI:', roi)
        roi_dir=os.path.join(data_dir, roi)
        model_dir = os.path.join(data_dir, roi, 'Models')
        os.chdir(roi_dir)

        # configure the covariates to use. Change *_bspline_* to *_int_*
        cov_file_tr=os.path.join(roi_dir, 'cov_bspline_tr.txt')

        # load train & test response files
        resp_file_tr=os.path.join(roi_dir, 'resp_tr.txt')

        # run a basic model on the training dataset and store the predicted response (yhat_tr), the variance of the
        # predicted response (s2_tr), the model parameters (nm), the  Zscores for the train data, and other
        #various metrics (metrics_tr)
        yhat_tr, s2_tr, nm_tr, Z_tr, metrics_tr = estimate(cov_file_tr, resp_file_tr, testresp=resp_file_tr,
                                                    testcov=cov_file_tr, alg='blr', optimizer='powell',
                                                    savemodel=True, saveoutput=False,standardize=False)
        Rho_tr=metrics_tr['Rho']
        EV_tr=metrics_tr['EXPV']

        if show_plots:
            #plot y versus y hat
            plot_y_v_yhat(cov_file_tr, resp_file_tr, yhat_tr, 'Training Data', struct_var, roi, Rho_tr, EV_tr)

        #create dummy design matrices
        dummy_cov_file_path_female, dummy_cov_file_path_male = \
            create_dummy_design_matrix(struct_var, agemin, agemax, cov_file_tr, spline_order, spline_knots, filepath)

        #compute splines and superimpose on data. Show on screen or save to file depending on show_plots value.
        plot_data_with_spline('Training Data', struct_var, cov_file_tr, resp_file_tr, dummy_cov_file_path_female,
                              dummy_cov_file_path_male, model_dir, roi, show_plots, filepath)

    plt.show()
    return data_dir, agemin, agemax

def calculate_avg_brain_age_acceleration_apply_model(roi_ids, desc_string, all_datav2, struct_var, show_plots, model_dir, spline_order,
                                                     spline_knots, working_dir, agemin, agemax, num_permute, permute, shuffnum):
    #############################  Apply Normative Model to Post-COVID Data ####################

    # create a shuffle stratified by age group
    if permute and shuffnum == 0:
        # for permutation testing
        list_to_shuffle = all_datav2['sex'].to_list()
        list_for_stratify = all_datav2['age']
        unique_values = set(list_for_stratify)
        # shuffle this list 100 times and save to list of lists
        list_of_shuffled_sex = []
        for _ in range(num_permute):
            shuffled_sex = []
            for value in unique_values:
                indices = [i for i, v in enumerate(list_for_stratify) if v == value]
                shuffled_indices = random.sample(indices, len(indices))
                for index in shuffled_indices:
                    shuffled_sex.append(list_to_shuffle[index])
            list_of_shuffled_sex.append(shuffled_sex)
            mystop = 1
        # save list of lists to file
        write_list_of_lists(list_of_shuffled_sex, f'{working_dir}/sexes_permuted.txt')

    if permute:
        list_of_shuffled_sex = read_list_of_lists(f'{working_dir}/sexes_permuted.txt')
        shuffled_sex = list_of_shuffled_sex[shuffnum]

        # Reorder the 'sex' column based on the new_order list
        all_datav2.loc[:, 'sex'] = shuffled_sex

    # Write number of each unique values to screen
    for index, value in all_datav2['sex'].value_counts().items():
        print(f'Number of values {index} is {value}')

    # separate the brain features (response variables) and predictors (age, gender) in to separate dataframes
    all_datav2_features = all_datav2.loc[:, roi_ids]
    all_datav2_covariates = all_datav2[['agedays', 'sex']]

    # average cortical thickness across all regions for each subject
    all_datav2_features = all_datav2_features.mean(axis=1).to_frame()
    all_datav2_features.rename(columns={0: 'avgcortthick'}, inplace=True)

    #make file diretories for output
    makenewdir('{}/avgct_{}_predict_files/'.format(working_dir, desc_string))
    makenewdir('{}/avgct_{}_predict_files/{}'.format(working_dir, desc_string, struct_var))
    makenewdir('{}/avgct_{}_predict_files/{}/plots'.format(working_dir, desc_string, struct_var))
    makenewdir('{}/avgct_{}_predict_files/{}/ROI_models'.format(working_dir, desc_string, struct_var))
    makenewdir('{}/avgct_{}_predict_files/{}/covariate_files'.format(working_dir, desc_string, struct_var))
    makenewdir('{}/avgct_{}_predict_files/{}/response_files'.format(working_dir, desc_string, struct_var))

    roi_ids = ['avgcortthick']

    #specify paths
    out_dir = ('{}/avgct_{}_predict_files/{}/ROI_models/'
               .format(working_dir, desc_string, struct_var))
    #  this path is where ROI_models folders are located
    predict_files_dir = ('{}/avgct_{}_predict_files/{}/ROI_models/'
                .format(working_dir, desc_string, struct_var))

    ##########
    # Create output directories for each region and place covariate and response files for that region in  each directory
    ##########
    for c in all_datav2_features.columns:
        all_datav2_features[c].to_csv(f'{working_dir}/resp_te_'+c+'.txt', header=False, index=False)
        all_datav2_covariates.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
        all_datav2_features.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

    for i in ['avgcortthick']:
        roidirname = ('{}/avgct_{}_predict_files/{}/ROI_models/{}'
                      .format(working_dir, desc_string, struct_var, i))
        makenewdir(roidirname)
        resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
        resp_te_filepath = roidirname + '/resp_te.txt'
        shutil.copyfile(resp_te_filename, resp_te_filepath)
        cov_te_filepath = roidirname + '/cov_te.txt'
        shutil.copyfile(f"{working_dir}/cov_te.txt", cov_te_filepath)

    movefiles("{}/resp_*.txt", "{}/avgct_{}_predict_files/{}/response_files/"
              .format(working_dir, working_dir, desc_string, struct_var))
    movefiles("{}/cov_t*.txt", "{}/avgct_{}_predict_files/{}/covariate_files/"
              .format(working_dir, working_dir, desc_string, struct_var))

    # Create Design Matrix and add in spline basis and intercept
    create_design_matrix('test', agemin, agemax, spline_order, spline_knots, roi_ids, out_dir)

    # Create dataframe to store Zscores
    Z_time2 = pd.DataFrame()
    Z_time2['participant_id'] = all_datav2['participant_id'].copy()
    Z_time2.reset_index(inplace=True, drop = True)

    ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

    #create design matrices for all regions and save files in respective directories
    create_design_matrix('test', agemin, agemax, spline_order, spline_knots, roi_ids, predict_files_dir)

    for roi in roi_ids:
        print('Running ROI:', roi)
        roi_dir=os.path.join(predict_files_dir, roi)
        model_dir = os.path.join(model_dir, roi, 'Models')
        os.chdir(roi_dir)

        # configure the covariates to use.
        cov_file_te=os.path.join(roi_dir, 'cov_bspline_te.txt')

        # load test response files
        resp_file_te=os.path.join(roi_dir, 'resp_te.txt')

        # make predictions
        yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

        Z_time2[roi] = Z

        #create dummy design matrices
        dummy_cov_file_path_female, dummy_cov_file_path_male = \
            create_dummy_design_matrix(struct_var, agemin, agemax, cov_file_te, spline_order, spline_knots, working_dir)

        #calculate brain age acceleration for each region
        mean_agediff_f, mean_agediff_m = calculate_age_acceleration(struct_var, roi_dir, yhat_te, model_dir, roi,
                                                        dummy_cov_file_path_female, dummy_cov_file_path_male, plotgap=1)


    if not permute:

        plot_data_with_spline('Postcovid (Test) Data ', struct_var, cov_file_te, resp_file_te, dummy_cov_file_path_female,
                                  dummy_cov_file_path_male, model_dir, roi, show_plots, working_dir)

        Z_time2.to_csv('{}/avgct_{}_predict_files/{}/Z_scores_by_region_postcovid_testset_avgct.txt'
                                .format(working_dir, desc_string, struct_var), index=False)


        plot_and_compute_zcores_by_gender(Z_time2, struct_var, roi_ids)
        plt.show()

    return mean_agediff_f, mean_agediff_m
