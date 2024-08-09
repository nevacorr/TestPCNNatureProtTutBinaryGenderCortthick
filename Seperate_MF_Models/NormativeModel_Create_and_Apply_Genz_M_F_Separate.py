#####
# This program implements the bayesian linear regression normative modeling outlined by Rutherford et al.
# NatureProtocols 2022 (https://doi.org/10.1038/s41596-022-00696-5). Here the modeling is applied to
# adolescent cortica1 thickness data collected at two time points (before and after the COVID lockdowns).
# This program creates models of cortical thickness change between 9 and 17 years of age for our pre-COVID data and
# stores these models to be applied in another script (Apply_Normative_Model_to_Genz_Time2.py).
# This program performs the modeling separately for male and females to allow for interaction between the genders.
# to the post-COVID data.
# Author: Neva M. Corrigan
######

import pandas as pd
import os
from Utility_Functions_MF_Separate import plot_age_acceleration
from make_time1_normative_model_MF_separate import make_time1_normative_model
from apply_normative_model_time2_MFseparate import apply_normative_model_time2
from plot_z_scores_MFseparate import plot_and_compute_zcores_by_gender
from calculate_avg_brain_age_acceleration_one_gender import calculate_avg_brain_age_acceleration_one_gender_make_model
from calculate_avg_brain_age_acceleration_one_gender import calculate_avg_brain_age_acceleration_one_gender_apply_model
from calculate_avg_brain_age_acceleration_bootstrap_one_gender import calculate_avg_brain_age_acceleration_one_gender_apply_model_bootstrap

orig_struct_var = 'cortthick'
show_plots = 0          #set to 1 to show training and test data ymvs yhat and spline fit plots.
show_nsubject_plots = 0 #set to 1 to plot number of subjects used in analysis, for each age and gender
spline_order = 1        # order of spline to use for model
spline_knots = 2        # number of knots in spline to use in model
nbootstrap = 1000         #number of bootstrap to use in calculating confidence intervals for age accelaration separately by sex

run_make_norm_model = 1
run_apply_norm_model = 1
calc_brain_age_acc = 1
calc_CI_age_acc_bootstrap = 1

orig_data_dir = '/home/toddr/neva/PycharmProjects/TestPCNNatureProtTutBinaryGenderCortthick'
working_dir = os.getcwd()

ageacc_from_bootstraps = {}
male = pd.DataFrame(columns=['mean_agediff'])
female = pd.DataFrame(columns=['mean_agediff'])
Z_time1 = {}
Z_time2 = {}
mean_agediff = {}

for gender in ['male', 'female']:

    if run_make_norm_model:

        Z_time1[gender] = make_time1_normative_model(gender, orig_struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                               orig_data_dir, working_dir)

        Z_time1[gender].drop(columns=['subject_id_test'], inplace=True)

    if run_apply_norm_model:

        Z_time2[gender] = apply_normative_model_time2(gender, orig_struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                orig_data_dir, working_dir)

    if calc_brain_age_acc:

        calculate_avg_brain_age_acceleration_one_gender_make_model(gender, orig_struct_var, show_nsubject_plots, show_plots,
                                                                   spline_order, spline_knots, orig_data_dir, working_dir)

        mean_agediff[gender] = calculate_avg_brain_age_acceleration_one_gender_apply_model(gender, orig_struct_var, show_nsubject_plots, show_plots,
                                                               spline_order, spline_knots, orig_data_dir, working_dir, num_permute=0, permute=False, shuffnum=0)

    if calc_CI_age_acc_bootstrap:

        ageacc_from_bootstraps[gender] = calculate_avg_brain_age_acceleration_one_gender_apply_model_bootstrap(gender, orig_struct_var, show_nsubject_plots, show_plots,
                                                           spline_order, spline_knots, orig_data_dir, working_dir, nbootstrap)
        # Write age acceleration from bootstrapping to file
        with open(f"{working_dir}/ageacceleration_dictionary {nbootstrap} bootstraps.txt", 'w') as f:
            for key, value in ageacc_from_bootstraps.items():
                f.write('%s:%s\n' % (key, value))

if calc_CI_age_acc_bootstrap:
    plot_age_acceleration(working_dir, nbootstrap, mean_agediff)

if run_apply_norm_model:
    Z_time2_male = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                               .format(working_dir, 'cortthick_male'))
    Z_time2_female = pd.read_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                                 .format(working_dir, 'cortthick_female'))

    Z_time2_male.to_csv(f'{working_dir}/predict_files/Z_time2_male.csv', index=False)
    Z_time2_female.to_csv(f'{working_dir}/predict_files/Z_time2_female.csv', index=False)

    Z_time2['male'] = Z_time2_male
    Z_time2['female'] = Z_time2_female

    plot_and_compute_zcores_by_gender(orig_struct_var, Z_time2, working_dir)

