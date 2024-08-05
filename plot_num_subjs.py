##########
# Plots number of subjects per age and gender
##########

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
def plot_num_subjs(df, title, struct_var, timept, path):
    sns.set_style(style='white')
    g = sns.catplot(x="age", hue="sex", data=df, kind="count", legend=False, palette=sns.color_palette(['green', 'blue']))
    g.fig.suptitle(title, fontsize=20)
    g.fig.subplots_adjust(top=0.85) # adjust the Figure
    g.ax.set_xlabel("Age", fontsize=18)
    g.ax.set_ylabel("Number of Subjects", fontsize=18)
    g.ax.tick_params(axis='x', labelsize=16)
    g.ax.tick_params(axis='y', labelsize=16)
    hue_labels = ['female', 'male']
    g.add_legend(legend_data={
        key: value for key, value in zip(hue_labels, g._legend_data.values())
        }, fontsize=18)
    g.ax.set(yticks=np.arange(0,20,2))
    plt.show(block=False)
    plt.savefig('{}/data/{}/plots/NumSubjects_{}'.format(path, struct_var, timept))
