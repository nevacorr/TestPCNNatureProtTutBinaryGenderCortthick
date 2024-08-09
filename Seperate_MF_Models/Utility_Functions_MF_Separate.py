#########
# This file contains a number of functions utilized in implementing normative modeling
##########
import os
import numpy as np
from matplotlib import pyplot as plt
from pcntoolkit.normative import predict
import pandas as pd
import seaborn as sns
import shutil
import glob
from pcntoolkit.util.utils import create_bspline_basis
from matplotlib.colors import ListedColormap
from scipy import stats
import ast

def makenewdir(path):
    isExist = os.path.exists(path)
    if isExist is False:
        os.mkdir(path)
        print('made directory {}'.format(path))

def movefiles(pattern, folder):
    files = glob.glob(pattern)
    for file in files:
        file_name = os.path.basename(file)
        shutil.move(file, folder + file_name)
        print('moved:', file)

def create_design_matrix_one_gender(datatype, agemin, agemax, spline_order, spline_knots, roi_ids, data_dir):
    B = create_bspline_basis(agemin, agemax, p=spline_order, nknots=spline_knots)
    for roi in roi_ids:
        print('Creating basis expansion for ROI:', roi)
        roi_dir = os.path.join(data_dir, roi)
        os.chdir(roi_dir)
        # Create output dir
        os.makedirs(os.path.join(roi_dir, 'blr'), exist_ok=True)

        # Load train & test covariate data matrices
        if datatype == 'train':
            X = np.loadtxt(os.path.join(roi_dir, 'cov_tr.txt'))
        elif datatype == 'test':
            f=os.listdir(roi_dir)
            p = os.path.join(roi_dir, 'cov_te.txt')
            tmp=np.loadtxt(p)
            X = np.loadtxt(os.path.join(roi_dir, 'cov_te.txt'))

        # Add intercept column
        X = np.vstack((X, np.ones(len(X)))).T

        if datatype == 'train':
            np.savetxt(os.path.join(roi_dir, 'cov_int_tr.txt'), X)
        elif datatype == 'test':
            np.savetxt(os.path.join(roi_dir, 'cov_int_te.txt'), X)

        # Create Bspline basis set
        # This creates a numpy array called Phi by applying function B to each element of the first column of X
        Phi = np.array([B(i) for i in X[:, 0]])
        X = np.concatenate((X, Phi), axis=1)
        if datatype == 'train':
            np.savetxt(os.path.join(roi_dir, 'cov_bspline_tr.txt'), X)
        elif datatype == 'test':
            np.savetxt(os.path.join(roi_dir, 'cov_bspline_te.txt'), X)

# This function creates a dummy design matrix for plotting of spline function
def create_dummy_design_matrix_one_gender(struct_var, agemin, agemax, cov_file, spline_order, spline_knots, path):

    # Make dummy test data covariate file starting with a column for age
    dummy_cov = np.linspace(agemin, agemax, num=1000)
    ones = np.ones((1, dummy_cov.shape[0]))

    # Add a column for intercept
    dummy_cov_final = np.vstack((dummy_cov, ones)).T

    # Create spline features and add them to predictor dataframe
    BAll = create_bspline_basis(agemin, agemax, p=spline_order, nknots=spline_knots)
    Phidummy = np.array([BAll(i) for i in dummy_cov_final[:, 0]])
    dummy_cov_final = np.concatenate((dummy_cov_final, Phidummy), axis=1)

    # Write these new created predictor variables with spline and response variable to file
    dummy_cov_file_path = os.path.join(path, 'cov_file_dummy.txt')
    np.savetxt(dummy_cov_file_path, dummy_cov_final)
    return dummy_cov_file_path


# This function plots  data with spline model superimposed, for both male and females
def plot_data_with_spline_one_gender(gender, datastr, struct_var, cov_file, resp_file, dummy_cov_file_path, model_dir, roi,
                                     showplots, working_dir):

    output = predict(dummy_cov_file_path, respfile=None, alg='blr', model_path=model_dir)

    yhat_predict_dummy=output[0]

    # Load real data predictor variables for region
    X = np.loadtxt(cov_file)
    # Load real data response variables for region
    y = np.loadtxt(resp_file)

    # Create dataframes for plotting with seaborn facetgrid objects
    dummy_cov = np.loadtxt(dummy_cov_file_path)
    df_origdata = pd.DataFrame(data=X[:, 0], columns=['Age in Days'])
    df_origdata[struct_var] = y.tolist()
    df_origdata['Age in Days'] = df_origdata['Age in Days'] / 365.25
    df_estspline = pd.DataFrame(data=dummy_cov[:, 0].tolist(),columns=['Age in Days'])
    df_estspline['Age in Days'] = df_estspline['Age in Days'] / 365.25
    tmp = np.array(yhat_predict_dummy.tolist(), dtype=float)
    df_estspline[struct_var] = tmp
    df_estspline = df_estspline.drop(index=df_estspline.iloc[999].name).reset_index(drop=True)

    # PLot figure
    fig=plt.figure()
    if gender == 'female':
        color = 'green'
    else:
        color = 'blue'
    sns.lineplot(data=df_estspline, x='Age in Days', y=struct_var, color=color, legend=False)
    sns.scatterplot(data=df_origdata, x='Age in Days', y=struct_var, color=color)
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    plt.title(datastr +' ' + struct_var +  ' vs. Age\n' + roi.replace(struct_var+'-', ''))
    plt.xlabel('Age')
    plt.ylabel(datastr + struct_var)
    if showplots == 1:
        if datastr == 'Training Data':
            plt.show(block=False)
        else:
            plt.show()
    else:
        plt.savefig('{}/data/{}/plots/{}_vs_age_withsplinefit_{}_{}'
                .format(working_dir, struct_var, struct_var, roi.replace(struct_var+'-', ''), datastr))
        plt.close(fig)
    if datastr == 'Training Data':
        splinemodel_fname = f'{working_dir}/data/{struct_var}/plots/spline_model_{datastr}_{roi}_{gender}.csv'
        origdata_fname = f'{working_dir}/data/{struct_var}/plots/datapoints_{datastr}_{roi}_{gender}.csv'
        df_estspline.to_csv(splinemodel_fname)
        df_origdata.to_csv(origdata_fname)

    # Write model to file if training set so male and female data and models can be viewed on same plot
    if datastr == 'Training Data':
        splinemodel_fname = f'{working_dir}/data/{struct_var}/plots/spline_model_{datastr}_{roi}_{gender}.csv'
        df_estspline.to_csv(splinemodel_fname)
        origdata_fname = f'{working_dir}/data/{struct_var}/plots/datapoints_{datastr}_{roi}_{gender}.csv'
    # Write actual data points to file for this data set
    if datastr == 'Postcovid (Test) Data ':
        origdata_fname = f'{working_dir}/predict_files/{struct_var}/plots/datapoints_{datastr}_{roi}_{gender}.csv'
    if datastr == 'Validation Data':
        origdata_fname = f'{working_dir}/data/{struct_var}/plots/datapoints_{datastr}_{roi}_{gender}.csv'
    df_origdata.to_csv(origdata_fname)
    mystop=1

def plot_y_v_yhat_one_gender(gender, cov_file, resp_file, yhat, typestring, struct_var, roi, Rho, EV):
    cov_data = np.loadtxt(cov_file)
    y = np.loadtxt(resp_file).reshape(-1,1)
    dfp = pd.DataFrame()
    y=y.flatten()
    dfp['y'] = y
    dfp['yhat'] = yhat
    print(dfp.dtypes)
    fig = plt.figure()
    if gender == 'female':
        color='green'
    else:
        color='blue'

    sns.scatterplot(data=dfp, x='y', y='yhat', color=color)
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    plt.title(typestring + ' ' + struct_var + ' vs. estimate\n'
              + roi +' EV=' + '{:.4}'.format(str(EV.item())) + ' Rho=' + '{:.4}'.format(str(Rho.item())))
    plt.xlabel(typestring + ' ' + struct_var)
    plt.ylabel(struct_var + ' estimate on ' + typestring)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red')  # plots line y = x
    plt.show(block=False)

def barplot_performance_values(struct_var, metric, df, spline_order, spline_knots, datastr, path, gender):
    colors = ['blue' if 'lh' in x else 'green' for x in df.ROI]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.barplot(x=df[metric], y=df['ROI'], hue=df['ROI'], orient='h', palette=colors, legend=False)
    plt.subplots_adjust(left=0.4)
    plt.subplots_adjust(top=0.93)
    plt.subplots_adjust(bottom=0.05)
    ax.set_title('Test Set ' + metric + ' for All Brain Regions' + ' ' + gender)
    plt.show(block=False)
    plt.savefig(
        '{}/data/{}/plots/{}_{}_for_all_regions_splineorder{}, splineknots{}_{}.png'
        .format(path, struct_var, datastr, metric, spline_order, spline_knots, gender))
    #plt.close(fig)

def write_ages_to_file_by_gender(wdir, agemin, agemax, struct_var, gender):
    with open("{}/agemin_agemax_Xtrain_{}.txt".format(wdir, struct_var, gender), "w") as file:
        # Write the values to the file
        file.write(str(agemin) + "\n")
        file.write(str(agemax) + "\n")

def read_ages_from_file(wdir, struct_var):
    # Open the file in read mode
    with open("{}/agemin_agemax_Xtrain_{}.txt".format(wdir, struct_var), "r") as file:
        # Read all lines from the file
        lines = file.readlines()
    # Extract the values from the lines
    agemin = int(lines[0].strip())
    agemax = int(lines[1].strip())
    return agemin, agemax

def write_list_to_file(mylist, filepath):
   with open(filepath, 'w') as file:
       for item in mylist:
           file.write(item + '\n')

def fit_regression_model_dummy_data(model_dir, dummy_cov_file_path):
    # create dummy data to find equation for linear regression fit between age and structvar
    dummy_predictors = pd.read_csv(dummy_cov_file_path, delim_whitespace=True, header=None)
    dummy_ages = dummy_predictors.iloc[:, 0]

    # calculate predicted values for dummy covariates for male and female
    output = predict(dummy_cov_file_path, respfile=None, alg='blr', model_path=model_dir)

    yhat_predict_dummy = output[0]

    # remove last element of age and output arrays
    yhat_predict_dummy = np.delete(yhat_predict_dummy, -1)
    dummy_ages = np.delete(dummy_ages.to_numpy(), -1)

    # find slope and intercept of lines
    slope, intercept, rvalue, pvalue, std_error = stats.linregress(dummy_ages, yhat_predict_dummy)

    # #plot dummy data with fit
    # plt.figure()
    # plt.plot(dummy_ages, yhat_predict_dummy, 'og', markersize=3, markerfacecolor='None')
    # plt.plot(dummy_ages, slope*dummy_ages+intercept, '-k', linewidth=1)
    # plt.show()

    return slope, intercept

def plot_age_acceleration(working_dir, nbootstrap, mean_agediff):
    # Initialize an empty dictionary
    ageacc_from_bootstraps = {}

    # Open the file with bootstrap results for age acceleration for both genders for reading
    with open(f"{working_dir}/ageacceleration_dictionary {nbootstrap} bootstraps.txt", 'r') as f:
        # Iterate over each line in the file
        for line in f:
            # Split the line into key-value pairs using the colon (:) as the separator
            key, value = line.strip().split(':')
            # Convert the value to the appropriate data type if needed (e.g., float)
            # Add the key-value pair to the dictionary
            ageacc_from_bootstraps[key] = value

    # Convert age acceleration dictionoary to series for each gender
    female_acc = pd.Series(ast.literal_eval(ageacc_from_bootstraps['female']))
    male_acc = pd.Series(ast.literal_eval(ageacc_from_bootstraps['male']))

    # Sort the series
    female_acc.sort_values(inplace=True)
    female_acc.reset_index(inplace=True, drop=True)
    male_acc.sort_values(inplace=True)
    male_acc.reset_index(inplace=True, drop=True)

    female_CI = np.percentile(female_acc, (2.5, 97.5), method='closest_observation')
    male_CI = np.percentile(male_acc, (2.5, 97.5), method='closest_observation')

    # Plot mean age acceleration with confidence intervals for males and females on same plots
    female_errors = (mean_agediff['female'] - female_CI[0], female_CI[1]-mean_agediff['female'])
    male_errors = (mean_agediff['male'] - male_CI[0], male_CI[1] - mean_agediff['male'])

    plt.figure(figsize=(4, 6))
    plt.errorbar(0.4, mean_agediff['female'],
                 yerr=[[female_errors[0]], [female_errors[1]]],
                 color='crimson', marker='o')
    plt.errorbar(0.2, mean_agediff['male'],
                 yerr=[[male_errors[0]], [male_errors[1]]],
                 color='blue', marker='o')

    plt.xlim([0, 0.6])
    plt.title('Age Acceleration By Sex\nwith 95% Confidence Interval')
    plt.xlabel('Sex', fontsize=12)
    plt.ylabel('Age Acceleration (years)', fontsize=12)
    plt.xticks([0.2, 0.4], labels=['Male', 'Female'], fontsize=12)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.show()
    mystop=1

def plot_age_acceleration_by_subject(y_yhat_df, gender, working_dir, struct_var):
    # Load file with age and predicted age
    y_yhat_df = pd.read_csv(f'{working_dir}/predict_files/avgct_{struct_var}/age and predicted age postcovid_test_data_{gender}.csv')

    # Load model mapping between cortical thickness and age
    model_mapping = pd.read_csv(f'{working_dir}/data/avgct_cortthick_{gender}/plots/spline_model_Training Data_avgcortthick_{gender}.csv')
    model_mapping.drop(columns=['Unnamed: 0'], inplace=True)
    model_mapping['Age in Days'] = model_mapping['Age in Days']*365.25

    # For every post-covid subjects, calculate what predicted age would be based on actual cortical thickness for that subject
    age_acceleration = []
    plot_df = pd.DataFrame()
    for val in range(y_yhat_df.shape[0]):
        index_match = model_mapping[f'avgct_cortthick_{gender}'].sub(y_yhat_df.loc[val, 'actual_avgcortthick']).abs().idxmin()
        predicted_age = model_mapping.loc[index_match, 'Age in Days']
        actual_age = y_yhat_df.loc[val, 'agedays']
        age_acceleration.append(predicted_age - actual_age)
        plot_df.loc[val, 'actual_age'] = actual_age/365.25
        plot_df.loc[val, 'predicted_age'] = predicted_age/365.25
        plot_df.loc[val, 'index'] = val
    avg_age_acceleration = (sum(age_acceleration) / len(age_acceleration))/365.25
    fig, axs = plt.subplots(2, figsize=(10, 8))
    axs[0].scatter(plot_df['index'], plot_df['actual_age'], color='red')
    axs[0].scatter(plot_df['index'], plot_df['predicted_age'], color='purple')
    axs[0].legend(['actual age', 'predicted age'], loc='lower left')
    axs[0].set_title(f'Actual Age and Predicted Age for all Post Covid Subjects {gender}')
    axs[0].set_xlabel('Subject Number')
    axs[0].set_ylabel('Age')
    axs[1].scatter(plot_df['index'], plot_df['predicted_age'] - plot_df['actual_age'], color = 'gray')
    axs[1].set_title(f'Predicted minus Actual Age for all Post Covid Subjects {gender}  Average = {avg_age_acceleration:.1f} years')
    axs[1].set_xlabel('Subject Number')
    axs[1].set_ylabel(f'Predicted minus Actual Age {gender} (years)')
    plt.show()
    mystop=1

def write_list_of_lists(data, file_path):
    # Write the list of lists to a file using list comprehension
    with open(file_path, 'w') as file:
        file.writelines([' '.join(str(num) for num in sublist) + '\n' for sublist in data])

def read_list_of_lists(file_path):
    # Read a list of lists from file using list comprehension
    read_data = []
    with open(file_path, 'r') as file:
        read_data = [[int(num) for num in line.split()] for line in file]
    return read_data