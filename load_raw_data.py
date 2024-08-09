import pandas as pd
import numpy as np
def load_raw_data(struct_var, visit, path):

    # Load corticl thickness data
    genz_data_combined = pd.read_csv(f'{path}/Adol_CortThick_data.csv')

    # remove rows with missing data
    genz_data_combined = genz_data_combined.dropna(ignore_index=True)

    # convert gender, agegroup and agemonths columns from float to int
    genz_data_combined['gender'] = genz_data_combined['gender'].astype('int64')
    genz_data_combined['agegroup'] = genz_data_combined['agegroup'].astype('int64')
    genz_data_combined['agemonths'] = genz_data_combined['agemonths'].astype('int64')
    genz_data_combined['agedays'] = genz_data_combined['agedays'].astype('int64')

    ##########
    # Load data quality measures
    ##########
    if visit == 1:
        euler = pd.read_csv(f'{path}/visit1_euler_numbers.csv', header=None)
    elif visit == 2:
        euler = pd.read_csv(f'{path}/visit2_euler_numbers.csv', header=None)

    ##########
    # Average left and right hemisphere euler numbers
    ##########
    euler['euler'] = (euler.iloc[:, 1] + euler.iloc[:, 2]) / 2.0
    euler.drop(euler.columns[[1, 2]], axis=1, inplace=True)
    euler['visit'] = visit
    euler['euler'] = euler['euler'].astype(int)
    # calculate median euler value
    median_euler = euler['euler'].median()
    # subtract median euler from all subjects, then multiply by -1 and take the square root
    euler['euler'] = euler['euler'] - median_euler
    euler['euler'] = euler['euler'] * -1
    euler['euler'] = np.sqrt(np.absolute(euler['euler']))

    ##########
    # Insert data quality measure into dataframe with brain data
    ##########
    euler.rename(columns={0: "subject"}, inplace=True)
    genz_data_combined = genz_data_combined.merge(euler, how='left', on=["subject", "visit"])
    #remove first column which has irrelevant values
    genz_data_combined.drop(genz_data_combined.columns[0], axis=1, inplace=True)
    # if struct_var is equal to gmv or cortthick exclude any rows where already transformed euler value is greater than or equal to 10
    keeprows = (genz_data_combined['euler'] < 10.00000) | (genz_data_combined['euler'].isna())
    genz_data_combined = genz_data_combined.loc[keeprows, :]

    ##########
    # Prepare covariate data
    # E########
    cov = pd.DataFrame()
    cov['participant_id'] = genz_data_combined['subject']
    cov['age'] = genz_data_combined['agegroup']
    cov['visit'] = genz_data_combined['visit']
    cov['sex'] = genz_data_combined['gender']

    return cov, genz_data_combined
