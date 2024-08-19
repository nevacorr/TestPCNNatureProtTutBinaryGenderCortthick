# Normative Modeling of Adolescent Cortical Thickness

This project implements Bayesian linear regression normative modeling according to the procedure outlined by Rutherford et al. in Nature Protocols 2022 (https://doi.org/10.1038/s41596-022-00696-5).
Here the modeling is applied to adolescent cortical thickness data collected at two time points (before and after the COVID-19 pandemic lockdowns) by Patricia Kuhl's laboratory at the University of Washington.
This project creates models based on pre-COVID data and applies these to the post-COVID data.

## Installing dependencies

To install the required software, please execute:

    pip install -r requirements.txt

## Input data:

**Adol_CortThick_data.csv** contains the data used in the analysis.

**visit1_subjects_used_to_create_normative_model_train_set_cortthick.txt** contains the list of subjects whose pre-COVID data were used for model training across all programs.

**visit2_all_subjects_used_in_test_set_cortthick.txt** contains the list of subjects whose post-COVID data were used to evaluate the effects of the COVID pandemic lockdowns across all programs.

**visit1_euler_numbers.csv** contains the euler numbers for the left and right hemispheres for each study subect at the pre-COVID timepoint.

**visit2_euler_numbers.csv** contains the euler numbers for the left and right hemispheres for each study subject at the post-COVID lockdown timepoint.

## Running the analysis

You can reproduce the results by running the following scripts in order:

1. **NormativeModelGenz_Time1.py** : run this file to generate the normative models for the pre-COVID data. This program saves the models to disk.

2. **Apply_Normative_Model_to_Genz_Time2.py** : run to apply the models to the post_COVID data. It utilizes the models produced by NormativeModelGenz_Time1.py.

3. **CalculateAvgBrainAgeAfterAveragingCorticalThicknesses.py** : run to compute average acceleration in cortical thickness observed in the post-COVID data. This code does not utilize the models generated by NormativeModelGenz_Time1.py or any output from Apply_Normative_Model_to_Genz_Time2.py. However, it does use the same train (pre-COVID) and test (post-COVID) subject cohorts that are  utilized by these other two programs.

4. **Calculate_Effect_Size_and_CI_using_Zscore.py** : run this to compute effect sizes and confidence intervals for effect sizes.

All other Python files are used to support the main python .py files listed
above. In these files, the phrases "time 1", "visit1", "training" or "train"
are used to refer to the pre-COVID data. The phrases "time 2", "visit 2" or
"test" refer to the post-COVID data, with one exception: within the
NormativeModelGenz_Time1.py file, "test" sometimes refers to a validation set
that is a subset of the training data. Comments within that file provide
clarification.

## Alternate analysis: Separate Models for Males and Females

You can reproduce the results in the alternate analysis which allows for interactions between the two sexes by creating separate normative models by running the following script which is in folder AlternateAnalysis:

1. **NormativeModel_Create_and_Apply_Genz_M_F_Separate** : run this file to generate the normative models from the pre-COVID data and apply them to the post-COVID data. This also computes the average acceleration in cortical thickness observed in the post-COVID data.

2. **Calculate_Effect_Size_and_CI_using_Zscore_MFseparate.py** : run this to compute effect sizes and confidence intervals for effect sizes.

These scripts uses functions contained in the other Python files located in this folder, plus some of the files in the main repository folder. 
