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
        shutil.move(file, folder +  file_name)
        print('moved:', file)

def create_design_matrix(datatype, agemin, agemax, spline_order, spline_knots, roi_ids, data_dir):
    B = create_bspline_basis(agemin, agemax, p=spline_order, nknots=spline_knots)
    for roi in roi_ids:
        print('Creating basis expansion for ROI:', roi)
        roi_dir = os.path.join(data_dir, roi)
        os.chdir(roi_dir)
        # create output dir
        os.makedirs(os.path.join(roi_dir, 'blr'), exist_ok=True)

        # load train & test covariate data matrices
        if datatype == 'train':
            X = np.loadtxt(os.path.join(roi_dir, 'cov_tr.txt'))
        elif datatype == 'test':
            X = np.loadtxt(os.path.join(roi_dir, 'cov_te.txt'))

        # add intercept column
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

        if datatype == 'train':
            np.savetxt(os.path.join(roi_dir, 'cov_int_tr.txt'), X)
        elif datatype == 'test':
            np.savetxt(os.path.join(roi_dir, 'cov_int_te.txt'), X)

        # create Bspline basis set
        # This creates a numpy array called Phi by applying function B to each element of the first column of X_tr
        Phi = np.array([B(i) for i in X[:, 0]])
        X = np.concatenate((X, Phi), axis=1)
        if datatype == 'train':
            np.savetxt(os.path.join(roi_dir, 'cov_bspline_tr.txt'), X)
        elif datatype == 'test':
            np.savetxt(os.path.join(roi_dir, 'cov_bspline_te.txt'), X)

#this function creates a dummy design matrix for plotting of spline function
def create_dummy_design_matrix(struct_var, agemin, agemax, cov_file, spline_order, spline_knots, path):
    # load predictor variables for region
    X = np.loadtxt(cov_file)

    # make dummy test data covariate file starting with a column for age
    dummy_cov = np.linspace(agemin, agemax, num=1000)
    ones = np.ones((dummy_cov.shape[0], 1))

    # add a column for gender for male and female data
    dummy_cov_female = np.concatenate((dummy_cov.reshape(-1, 1), ones * 0), axis=1)
    dummy_cov_male = np.concatenate((dummy_cov.reshape(-1, 1), ones), axis=1)

    #add a column for intercept
    dummy_cov_female = np.concatenate((dummy_cov_female, ones), axis=1)
    dummy_cov_male = np.concatenate((dummy_cov_male, ones), axis=1)

    # create spline features and add them to male and female predictor dataframes
    BAll = create_bspline_basis(agemin, agemax, p=spline_order, nknots=spline_knots)
    Phidummy_f = np.array([BAll(i) for i in dummy_cov_female[:, 0]])
    Phidummy_m = np.array([BAll(i) for i in dummy_cov_male[:, 0]])
    dummy_cov_female = np.concatenate((dummy_cov_female, Phidummy_f), axis=1)
    dummy_cov_male = np.concatenate((dummy_cov_male, Phidummy_m), axis=1)

    # write these new created predictor variables with spline and response variable to file
    dummy_cov_file_path_female = os.path.join(path, 'cov_file_dummy_female.txt')
    np.savetxt(dummy_cov_file_path_female, dummy_cov_female)
    dummy_cov_file_path_male = os.path.join(path, 'cov_file_dummy_male.txt')
    np.savetxt(dummy_cov_file_path_male, dummy_cov_male)
    return dummy_cov_file_path_female, dummy_cov_file_path_male


# this function plots  data with spline model superimposed, for both male and females
def plot_data_with_spline(datastr, struct_var, cov_file, resp_file, dummy_cov_file_path_female,
                              dummy_cov_file_path_male, model_dir, roi, showplots, working_dir):

    output_f = predict(dummy_cov_file_path_female, respfile=None, alg='blr', model_path=model_dir)

    output_m = predict(dummy_cov_file_path_male, respfile=None, alg='blr', model_path=model_dir)

    yhat_predict_dummy_m=output_m[0]
    yhat_predict_dummy_f=output_f[0]

    # load real data predictor variables for region
    X = np.loadtxt(cov_file)
    # load real data response variables for region
    y = np.loadtxt(resp_file)

    # create dataframes for plotting with seaborn facetgrid objects
    dummy_cov_female = np.loadtxt(dummy_cov_file_path_female)
    dummy_cov_male = np.loadtxt(dummy_cov_file_path_male)
    df_origdata = pd.DataFrame(data=X[:, 0:2], columns=['Age in Days', 'gender'])
    df_origdata[struct_var] = y.tolist()
    df_origdata['Age in Days'] = df_origdata['Age in Days'] / 365.25
    df_estspline = pd.DataFrame(data=dummy_cov_female[:, 0].tolist() + dummy_cov_male[:, 0].tolist(),
                                columns=['Age in Days'])
    df_estspline['Age in Days'] = df_estspline['Age in Days'] / 365.25
    df_estspline['gender'] = [0] * 1000 + [1] * 1000
    df_estspline['gender'] = df_estspline['gender'].astype('float')
    tmp = np.array(yhat_predict_dummy_f.tolist() + yhat_predict_dummy_m.tolist(), dtype=float)
    df_estspline[struct_var] = tmp
    df_estspline = df_estspline.drop(index=df_estspline.iloc[999].name).reset_index(drop=True)
    df_estspline = df_estspline.drop(index=df_estspline.iloc[1998].name)

    fig=plt.figure()
    colors = {1: 'blue', 0: 'green'}
    sns.lineplot(data=df_estspline, x='Age in Days', y=struct_var, hue='gender', palette=colors, legend=False)
    sns.scatterplot(data=df_origdata, x='Age in Days', y=struct_var, hue='gender', palette=colors)
    plt.legend(title='')
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    handles, labels = ax.get_legend_handles_labels()
    labels = ["female", "male"]
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

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

def plot_y_v_yhat(cov_file, resp_file, yhat, typestring, struct_var, roi, Rho, EV):
    cov_data = np.loadtxt(cov_file)
    gender = cov_data[:,1].reshape(-1,1)
    y = np.loadtxt(resp_file).reshape(-1,1)
    dfp = pd.DataFrame()
    gender=gender.flatten()
    y=y.flatten()
    yht=yhat.flatten()
    dfp['gender'] = gender
    dfp['y'] = y
    dfp['yhat'] = yhat
    print(dfp.dtypes)
    fig = plt.figure()
    colors = {1: 'blue', 0: 'green'}
    sns.scatterplot(data=dfp, x='y', y='yhat', hue='gender', palette=colors)
    ax = plt.gca()
    fig.subplots_adjust(right=0.82)
    handles, labels = ax.get_legend_handles_labels()
    labels = ["female", "male"]
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(typestring + ' ' + struct_var + ' vs. estimate\n'
              + roi +' EV=' + '{:.4}'.format(str(EV.item())) + ' Rho=' + '{:.4}'.format(str(Rho.item())))
    plt.xlabel(typestring + ' ' + struct_var)
    plt.ylabel(struct_var + ' estimate on ' + typestring)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red')  # plots line y = x
    plt.show(block=False)

def barplot_performance_values(struct_var, metric, df, spline_order, spline_knots, datastr, path):
    colors = ['blue' if 'lh' in x else 'green' for x in df.ROI]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.barplot(x=df[metric], y=df['ROI'], orient='h', palette=colors)
    plt.subplots_adjust(left=0.4)
    plt.subplots_adjust(top=0.93)
    plt.subplots_adjust(bottom=0.05)
    ax.set_title('Test Set ' + metric + ' for All Brain Regions')
    plt.show(block=False)
    plt.savefig(
        '{}/data/{}/plots/{}_{}_for_all_regions_splineorder{}, splineknots{}.png'
        .format(path, struct_var, datastr, metric, spline_order, spline_knots))
    #plt.close(fig)

def write_ages_to_file(agemin, agemax, struct_var):
    with open("agemin_agemax_Xtrain_{}.txt".format(struct_var), "w") as file:
        # Write the values to the file
        file.write(str(agemin) + "\n")
        file.write(str(agemax) + "\n")

def read_ages_from_file(struct_var):
    # Open the file in read mode
    with open("agemin_agemax_Xtrain_{}.txt".format(struct_var), "r") as file:
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

def plot_brain_age_gap_by_gender(brain_age_gap_df, model_type, include_gender):
    #input dataframe must have a 'gender' columns and an 'agediff' column
    #plot figure
    fig=plt.figure(figsize=(7,7))
    if include_gender ==0:
        #plot histograms
        mean_diff = brain_age_gap_df['agediff'].mean()
        sns.histplot(data=brain_age_gap_df, x='agediff')
        plt.title(
            f'{model_type} Distributions of difference between post-COVID\n lockdown age and actual age\n '
            f'mean diff = {mean_diff:.1f} years')
    elif include_gender == 1:
       # find mean of age_diff by gender
        means_by_gender = brain_age_gap_df.groupby('gender')['agediff'].mean()
        mean_diff_male = means_by_gender[1]
        mean_diff_female = means_by_gender[2]
        p = {'male': 'blue', 'female': 'orange'}
        brain_age_gap_df['gender'].replace({1: 'male', 2: 'female'}, inplace=True)
        #plot histograms
        sns.histplot(data=brain_age_gap_df, x='agediff', hue='gender', palette = p, element='step')
        plt.title(
            f'{model_type} Distributions of difference between post-COVID\n lockdown age and actual age by gender\n '
            f'mean diff male = {mean_diff_male:.1f} years mean diff female = {mean_diff_female:.1f} years')
    plt.xlabel('Predicted post-Covid lock down age  - actual age (years)')
    plt.ylabel('Number of subjects')
    plt.show(block=False)

def plotactual_age_vs_predicted_age(X_test_v2, y_test_v2, y_test_pred_v2, datatypestr, model_type, include_gender):
    if datatypestr == 'train':
        covid_str = 'pre-covid'
    elif datatypestr == 'test':
        covid_str = 'post-covid'
    plt.figure()
    ax=plt.axes()
    if include_gender ==0:
        sc = ax.scatter(y_test_v2, y_test_pred_v2)
    elif include_gender ==1:
        gen = X_test_v2['sex']
        cmap_custom = ListedColormap(['blue', 'orange'])
        sc = ax.scatter(y_test_v2, y_test_pred_v2, c=gen, cmap=cmap_custom)
        handles, labels = sc.legend_elements()
        labels=['male', 'female']
        ax.legend(handles, labels)
    ax.plot([y_test_v2.min(), y_test_v2.max()], [y_test_v2.min(), y_test_v2.max()], color='red')  # plots line y = x
    ax.set_xlabel('actual age (days)')
    ax.set_ylabel('predicted age (days)')
    ax.set_title(f'{model_type} predicted vs actual age for {covid_str} lockdown data')
    plt.show(block=False)

def fit_regression_model_dummy_data(model_dir, dummy_cov_file_path_female, dummy_cov_file_path_male):
    # create dummy data to find equation for linear regression fit between age and structvar
    dummy_predictors_f = pd.read_csv(dummy_cov_file_path_female, delim_whitespace=True, header=None)
    dummy_predictors_m = pd.read_csv(dummy_cov_file_path_male, delim_whitespace=True, header=None)
    dummy_ages_f = dummy_predictors_f.iloc[:, 0]
    dummy_ages_m = dummy_predictors_m.iloc[:, 0]

    # calculate predicted values for dummy covariates for male and female
    output_f = predict(dummy_cov_file_path_female, respfile=None, alg='blr', model_path=model_dir)
    output_m = predict(dummy_cov_file_path_male, respfile=None, alg='blr', model_path=model_dir)

    yhat_predict_dummy_f = output_f[0]
    yhat_predict_dummy_m = output_m[0]

    # remove last element of age and output arrays
    last_index = len(yhat_predict_dummy_f) - 1
    yhat_predict_dummy_f = np.delete(yhat_predict_dummy_f, -1)
    yhat_predict_dummy_m = np.delete(yhat_predict_dummy_m, -1)
    dummy_ages_f = np.delete(dummy_ages_f.to_numpy(), -1)
    dummy_ages_m = np.delete(dummy_ages_m.to_numpy(), -1)

    # find slope and intercept of lines
    slope_f, intercept_f, rvalue_f, pvalue_f, std_error_f = stats.linregress(dummy_ages_f, yhat_predict_dummy_f)
    slope_m, intercept_m, rvalue_m, pvalue_m, std_error_m = stats.linregress(dummy_ages_m, yhat_predict_dummy_m)

    # #plot dummy data with fit
    # plt.figure()
    # plt.plot(dummy_ages_f, yhat_predict_dummy_f, 'og', markersize=3, markerfacecolor='None')
    # plt.plot(dummy_ages_m, yhat_predict_dummy_m, 'ob', markersize=3, markerfacecolor='None')
    # plt.plot(dummy_ages_f, slope_f*dummy_ages_f+intercept_f, '-k', linewidth=1)
    # plt.plot(dummy_ages_m, slope_m*dummy_ages_f+intercept_m, '-k', linewidth=1)
    # # plt.title(roi)
    # plt.show(block=False)

    return slope_f, intercept_f, slope_m, intercept_m

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

def plot_age_acceleration(working_dir, nbootstrap, mean_agediff_f, mean_agediff_m):
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
    plt.figure(figsize=(4, 6))
    plt.errorbar(0.4, mean_agediff_f,
                 np.array(mean_agediff_f - female_CI[0], female_CI[1] - mean_agediff_f),
                 color='green', marker='o')
    plt.errorbar(0.2, mean_agediff_m,
                 np.array(mean_agediff_m - male_CI[0], male_CI[1] - mean_agediff_m),
                 color='blue', marker='o')

    plt.xlim([0, 0.6])
    plt.title('Age Acceleration By Sex\nwith 95% Confidence Interval')
    plt.xlabel('Sex', fontsize=12)
    plt.ylabel('Age Acceleration (years)', fontsize=12)
    plt.xticks([0.2, 0.4], labels=['Male', 'Female'], fontsize=12)
    plt.show(block=False)