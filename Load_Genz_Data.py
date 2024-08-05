# This function loads all adolescent visit 1 and visit 2 brain structural measures and behavior data,
# demographic measures, puberty measures and MRI data quality measures. It then only keeps the data
# for the visit of interest and returns the structural brain values for all regions, the covariates,
# and a list of all region names
##########

def load_genz_data(struct_var, visit, path):

    import pandas as pd
    from load_raw_data import load_raw_data

    cov, genz_data_combined = load_raw_data(struct_var, visit, path)

    # keep all rows from visit to keep in covariate dataframe
    cov = cov.loc[cov['visit'] == visit]

    # drop visit column
    cov.drop(columns='visit', inplace=True)

    ##########
    # Prepare brain data. Identify all columns with cortthick values
    ##########
    # make a list of columns of struct variable of interest
    struct_cols = [col for col in genz_data_combined.columns if struct_var + '-' in col]

    # create brain data dataframe with struct_var columns for visit
    brain_good = pd.DataFrame()
    brain_good['participant_id'] = genz_data_combined['subject']
    brain_good['visit'] = genz_data_combined['visit']
    brain_good['agedays'] = genz_data_combined['agedays']
    print(brain_good.shape)
    print(genz_data_combined.shape)
    brain_good[struct_cols] = genz_data_combined[struct_cols]

    # keep all rows from visit indicated
    brain_good = brain_good.loc[brain_good['visit'] == visit]
    # drop visit column
    brain_good.drop(columns='visit', inplace=True)

    # Check that subject rows align across covariate and brain dataframes
    # Make sure to use how = "inner" so that we only include subjects with data in both dataframes
    all_data = pd.merge(cov, brain_good, how='inner')
    # create a list of all the columns to run a normative model for
    roi_ids=all_data.columns.values.tolist()
    #remove subject info from list of brain regions to run normative model on
    remove_list = ['participant_id', 'age', 'sex', 'agedays']
    roi_ids = [i for i in roi_ids if i not in remove_list]

    return brain_good, all_data, roi_ids
