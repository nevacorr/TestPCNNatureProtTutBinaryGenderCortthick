
import numpy as np
from matplotlib import pyplot as plt
from pcntoolkit.normative import predict
import pandas as pd
import seaborn as sns

def plot_data_with_spline_rh_fusiform(datastr, struct_var, cov_file, resp_file, dummy_cov_file_path_female,
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

    #Make males listed first in legend
    #create a mapping of labels to handles
    label_to_handle = dict(zip(labels, handles))

    #reorder labels and handles
    ordered_handles = [label_to_handle['1.0'], label_to_handle['0.0']]

    #create legend
    ax.legend(ordered_handles, ['male', 'female'], loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    plt.title(datastr + 'Cortical Thickness vs Age\nRight Hemisphere Fusiform')
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Cortical Thickness (mm)', fontsize=12)
    plt.ylim(2.2, 3.15)
    plt.tight_layout()
    if showplots == 1:
        if datastr == 'Training Data':
            plt.show(block=False)
        else:
            plt.show()
    else:
        plt.savefig('{}/data/{}/plots/{}_vs_age_withsplinefit_{}_{}_fusiform_matchaxis_prepost'
                .format(working_dir, struct_var, struct_var, roi.replace(struct_var+'-', ''), datastr))
        plt.close(fig)
